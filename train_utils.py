import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from config import TrainConfig, default_train_config_path, load_train_config
from env import OrbitWarsEnv
from vector_env import SubprocVectorEnv
from features import TurnBatch, candidate_feature_dim, global_feature_dim, self_feature_dim
from game_types import PlanetState
from opponents import SelfPlayOpponent, build_opponent
from policy import PlanetPolicy
from algorithms.ppo import TransitionBatch, ppo_update, sample_actions

@dataclass(slots=True)
class StepGroup:
    indices: list[int]
    reward: float
    done: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(default_train_config_path()))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collect_rollout(
    envs: list[OrbitWarsEnv],
    batches: list[TurnBatch],
    policy: PlanetPolicy,
    cfg: TrainConfig,
    device: torch.device,
    next_seed: int,
) -> tuple[TransitionBatch, list[TurnBatch], int, dict[str, float]]:
    empty_candidate = (cfg.env.candidate_count, candidate_feature_dim())
    self_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    global_rows: list[np.ndarray] = []
    candidate_masks: list[np.ndarray] = []
    target_indices: list[int] = []
    log_probs: list[float] = []
    values: list[float] = []
    groups_per_env: list[list[StepGroup]] = [[] for _ in envs]
    episode_rewards: list[float] = []
    running_episode_rewards = [0.0 for _ in envs]

    for _ in range(cfg.ppo.rollout_steps):
        offsets = np.cumsum([0] + [batch.self_features.shape[0] for batch in batches[:-1]])
        merged = merge_batches(batches)
        row_values = np.zeros((merged.self_features.shape[0],), dtype=np.float32)
        if merged.self_features.shape[0] > 0:
            with torch.inference_mode():
                outputs = policy(
                    torch.from_numpy(merged.self_features).to(device),
                    torch.from_numpy(merged.candidate_features).to(device),
                    torch.from_numpy(merged.global_features).to(device),
                    torch.from_numpy(merged.candidate_mask).to(device).bool(),
                )
                sampled = sample_actions(outputs, deterministic=False)
                row_values = outputs.value.detach().cpu().numpy()
                sampled_target_index = sampled.target_index.detach().cpu().numpy()
                sampled_log_prob = sampled.log_prob.detach().cpu().numpy()
        else:
            sampled_target_index = np.zeros((0,), dtype=np.int64)
            sampled_log_prob = np.zeros((0,), dtype=np.float32)

        next_batches: list[TurnBatch] = []
        for env_idx, env in enumerate(envs):
            batch = batches[env_idx]
            start = int(offsets[env_idx])
            moves = []
            group_indices: list[int] = []
            for local_idx, context in enumerate(batch.contexts):
                global_idx = start + local_idx
                self_rows.append(batch.self_features[local_idx])
                candidate_rows.append(batch.candidate_features[local_idx])
                global_rows.append(batch.global_features[local_idx])
                candidate_masks.append(batch.candidate_mask[local_idx])
                values.append(float(row_values[global_idx]))
                tgt_idx = int(sampled_target_index[global_idx]) if batch.self_features.shape[0] > 0 else 0
                is_valid_send = (
                    tgt_idx > 0
                    and tgt_idx < len(context.candidate_ids)
                    and context.candidate_mask[tgt_idx]
                    and int(context.ship_counts[tgt_idx]) > 0
                )
                target_indices.append(tgt_idx)
                log_probs.append(float(sampled_log_prob[global_idx]) if batch.self_features.shape[0] > 0 else 0.0)
                group_indices.append(len(values) - 1)
                if not is_valid_send:
                    continue
                ships = int(context.ship_counts[tgt_idx])
                src_planet = find_planet(batch.state.planets, context.source_id)
                if src_planet is None or src_planet.ships < ships:
                    continue
                moves.append([context.source_id, float(context.target_angles[tgt_idx]), ships])
            result = env.step(moves)
            running_episode_rewards[env_idx] += float(result.reward)
            groups_per_env[env_idx].append(StepGroup(indices=group_indices, reward=float(result.reward), done=result.done))
            if result.done:
                episode_rewards.append(running_episode_rewards[env_idx])
                running_episode_rewards[env_idx] = 0.0
                next_seed += 1
                next_batch = env.reset(seed=next_seed)
            else:
                next_batch = result.batch
            next_batches.append(next_batch)
        batches = next_batches

    # Calculate Advantage
    returns: list[float] = [0.0] * len(values)
    advantages: list[float] = [0.0] * len(values)
    next_state_values = bootstrap_values(policy, batches, device)
    # # Calculate TD Advantage (High Variance)
    # for env_idx, groups in enumerate(groups_per_env):
    #     future_return = next_state_values[env_idx]
    #     for group in reversed(groups):
    #         future_return = group.reward + cfg.ppo.gamma * future_return * (1.0 - float(group.done))
    #         for idx in group.indices:
    #             returns[idx] = future_return
    #             advantages[idx] = future_return - values[idx]
    # Calculate Generalized Advantage Estimate (Best of both TD and MC, lower bias and variance)
    for env_idx, groups in enumerate(groups_per_env):
        last_gae_lam = 0.0
        next_value = next_state_values[env_idx]
        for group in reversed(groups):
            if not group.indices:
                # No actions were taken. Pass the reward, discounted value, and decayed GAE backward to the previous step.
                next_value = group.reward + cfg.ppo.gamma * next_value * (1.0 - float(group.done))
                last_gae_lam = cfg.ppo.gamma * cfg.ppo.lmbda * last_gae_lam * (1.0 - float(group.done))
                continue
            
            delta = group.reward + cfg.ppo.gamma * next_value * (1.0 - float(group.done)) - values[group.indices[0]]
            last_gae_lam = delta + cfg.ppo.gamma * cfg.ppo.lmbda * last_gae_lam * (1.0 - float(group.done))
            
            for idx in group.indices:
                advantages[idx] = last_gae_lam
                returns[idx] = last_gae_lam + values[idx]
                
            # The current state's value becomes the next_value for the previous step
            next_value = values[group.indices[0]]

    batch = TransitionBatch(
        self_features=torch.from_numpy(np.asarray(self_rows, dtype=np.float32).reshape(-1, self_feature_dim())),
        candidate_features=torch.from_numpy(
            np.asarray(candidate_rows, dtype=np.float32).reshape(-1, empty_candidate[0], empty_candidate[1])
        ),
        global_features=torch.from_numpy(np.asarray(global_rows, dtype=np.float32).reshape(-1, global_feature_dim())),
        candidate_mask=torch.from_numpy(np.asarray(candidate_masks, dtype=bool).reshape(-1, cfg.env.candidate_count)),
        target_index=torch.tensor(target_indices, dtype=torch.long),
        log_prob=torch.tensor(log_probs, dtype=torch.float32),
        returns=torch.tensor(returns, dtype=torch.float32),
        advantages=torch.tensor(advantages, dtype=torch.float32),
    )
    stats = {
        "episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "episodes_finished": float(len(episode_rewards)),
        "samples": float(len(values)),
    }
    return batch, batches, next_seed, stats


def bootstrap_values(policy: PlanetPolicy, batches: list[TurnBatch], device: torch.device) -> list[float]:
    merged = merge_batches(batches)
    if merged.self_features.shape[0] == 0:
        return [0.0 for _ in batches]
    offsets = np.cumsum([0] + [batch.self_features.shape[0] for batch in batches[:-1]])
    with torch.inference_mode():
        outputs = policy(
            torch.from_numpy(merged.self_features).to(device),
            torch.from_numpy(merged.candidate_features).to(device),
            torch.from_numpy(merged.global_features).to(device),
            torch.from_numpy(merged.candidate_mask).to(device).bool(),
        )
    values = outputs.value.detach().cpu().numpy()
    per_env = []
    for env_idx, batch in enumerate(batches):
        start = int(offsets[env_idx])
        count = batch.self_features.shape[0]
        per_env.append(0.0 if count == 0 else float(values[start : start + count].mean()))
    return per_env

def merge_batches(batches: list[TurnBatch]) -> TurnBatch:
    if not batches:
        raise ValueError("batches must not be empty")
    has_rows = any(batch.self_features.shape[0] > 0 for batch in batches)
    self_rows = (
        np.concatenate([batch.self_features for batch in batches], axis=0)
        if has_rows
        else np.zeros((0, self_feature_dim()), dtype=np.float32)
    )
    candidate_rows = (
        np.concatenate([batch.candidate_features for batch in batches], axis=0)
        if has_rows
        else np.zeros((0, batches[0].candidate_features.shape[1], candidate_feature_dim()), dtype=np.float32)
    )
    global_rows = (
        np.concatenate([batch.global_features for batch in batches], axis=0)
        if has_rows
        else np.zeros((0, global_feature_dim()), dtype=np.float32)
    )
    candidate_masks = (
        np.concatenate([batch.candidate_mask for batch in batches], axis=0)
        if has_rows
        else np.zeros((0, batches[0].candidate_mask.shape[1]), dtype=bool)
    )

    return TurnBatch(
        self_features=self_rows,
        candidate_features=candidate_rows,
        global_features=global_rows,
        candidate_mask=candidate_masks,
        contexts=[context for batch in batches for context in batch.contexts],
        state=batches[0].state,
    )

def save_checkpoint(
    save_dir: Path,
    run_name: str,
    update: int,
    policy: PlanetPolicy,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> None:
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "update": update,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        run_dir / "ckpt_last.pt",
    )
    torch.save(
        {
            "update": update,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        run_dir / f"ckpt_{update:06d}.pt",
    )

def find_planet(planets: list[PlanetState], planet_id: int) -> PlanetState | None:
    for planet in planets:
        if planet.id == planet_id:
            return planet
    return None


def build_single_env(idx: int, cfg: TrainConfig, device: torch.device) -> OrbitWarsEnv:
    opponent = build_opponent(cfg.opponent, cfg=cfg, device=device)
    return OrbitWarsEnv(cfg, opponent, env_index=idx)


def multiprocessing_collect_rollout(
    envs: SubprocVectorEnv,
    batches: list[TurnBatch],
    policy: PlanetPolicy,
    cfg: TrainConfig,
    device: torch.device,
    next_seed: int,
) -> tuple[TransitionBatch, list[TurnBatch], int, dict[str, float]]:
    empty_candidate = (cfg.env.candidate_count, candidate_feature_dim())
    self_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    global_rows: list[np.ndarray] = []
    candidate_masks: list[np.ndarray] = []
    target_indices: list[int] = []
    log_probs: list[float] = []
    values: list[float] = []

    num_envs = len(batches)
    groups_per_env: list[list[StepGroup]] = [[] for _ in range(num_envs)]
    episode_rewards: list[float] = []
    running_episode_rewards = [0.0 for _ in range(num_envs)]

    for _ in range(cfg.ppo.rollout_steps):
        # GET ACTIONS AND LOG PROBS FROM MODEL
        offsets = np.cumsum([0] + [batch.self_features.shape[0] for batch in batches[:-1]])
        merged = merge_batches(batches)
        row_values = np.zeros((merged.self_features.shape[0],), dtype=np.float32)
        if merged.self_features.shape[0] > 0:
            with torch.inference_mode():
                outputs = policy(
                    torch.from_numpy(merged.self_features).to(device),
                    torch.from_numpy(merged.candidate_features).to(device),
                    torch.from_numpy(merged.global_features).to(device),
                    torch.from_numpy(merged.candidate_mask).to(device).bool(),
                )
                sampled = sample_actions(outputs, deterministic=False)
                row_values = outputs.value.detach().cpu().numpy()
                sampled_target_index = sampled.target_index.detach().cpu().numpy()
                sampled_log_prob = sampled.log_prob.detach().cpu().numpy()
        else:
            sampled_target_index = np.zeros((0,), dtype=np.int64)
            sampled_log_prob = np.zeros((0,), dtype=np.float32)

        # BUILD ACTIONS FOR ALL ENVIROMENTS
        all_moves = []
        all_group_indices = []

        for env_idx in range(len(batches)):
            batch = batches[env_idx]
            start = int(offsets[env_idx])
            moves = []
            group_indices: list[int] = []
            
            for local_idx, context in enumerate(batch.contexts):
                global_idx = start + local_idx
                self_rows.append(batch.self_features[local_idx])
                candidate_rows.append(batch.candidate_features[local_idx])
                global_rows.append(batch.global_features[local_idx])
                candidate_masks.append(batch.candidate_mask[local_idx])
                values.append(float(row_values[global_idx]))
                
                tgt_idx = int(sampled_target_index[global_idx]) if batch.self_features.shape[0] > 0 else 0
                is_valid_send = (
                    tgt_idx > 0
                    and tgt_idx < len(context.candidate_ids)
                    and context.candidate_mask[tgt_idx]
                    and int(context.ship_counts[tgt_idx]) > 0
                )
                target_indices.append(tgt_idx)
                log_probs.append(float(sampled_log_prob[global_idx]) if batch.self_features.shape[0] > 0 else 0.0)
                group_indices.append(len(values) - 1)
                
                if not is_valid_send:
                    continue
                ships = int(context.ship_counts[tgt_idx])
                src_planet = find_planet(batch.state.planets, context.source_id)
                if src_planet is None or src_planet.ships < ships:
                    continue
                moves.append([context.source_id, float(context.target_angles[tgt_idx]), ships])
            
            all_moves.append(moves)
            all_group_indices.append(group_indices)

        # STEP ENVIROMENTS
        results = envs.step(all_moves)

        # PROCESS RESULTS & ASYNC RESETS
        next_batches: list[TurnBatch] = []
        for env_idx, result in enumerate(results):
            running_episode_rewards[env_idx] += float(result.reward)
            groups_per_env[env_idx].append(
                StepGroup(indices=all_group_indices[env_idx], reward=float(result.reward), done=result.done)
            )
            
            if result.done:
                episode_rewards.append(running_episode_rewards[env_idx])
                running_episode_rewards[env_idx] = 0.0
                next_seed += 1
                next_batch = envs.reset_one(env_idx, seed=next_seed)    
            else:
                next_batch = result.batch
                
            next_batches.append(next_batch)
            
        batches = next_batches

    # CALCULATE ADVANTAGE
    returns: list[float] = [0.0] * len(values)
    advantages: list[float] = [0.0] * len(values)
    next_state_values = bootstrap_values(policy, batches, device)
    # # Calculate TD Advantage (High Variance)
    # for env_idx, groups in enumerate(groups_per_env):
    #     future_return = next_state_values[env_idx]
    #     for group in reversed(groups):
    #         future_return = group.reward + cfg.ppo.gamma * future_return * (1.0 - float(group.done))
    #         for idx in group.indices:
    #             returns[idx] = future_return
    #             advantages[idx] = future_return - values[idx]
    # Calculate Generalized Advantage Estimate (Best of both TD and MC, lower bias and variance)
    for env_idx, groups in enumerate(groups_per_env):
        last_gae_lam = 0.0
        next_value = next_state_values[env_idx]
        for group in reversed(groups):
            if not group.indices:
                # No actions were taken. Pass the reward, discounted value, and decayed GAE backward to the previous step.
                next_value = group.reward + cfg.ppo.gamma * next_value * (1.0 - float(group.done))
                last_gae_lam = cfg.ppo.gamma * cfg.ppo.lmbda * last_gae_lam * (1.0 - float(group.done))
                continue
            
            delta = group.reward + cfg.ppo.gamma * next_value * (1.0 - float(group.done)) - values[group.indices[0]]
            last_gae_lam = delta + cfg.ppo.gamma * cfg.ppo.lmbda * last_gae_lam * (1.0 - float(group.done))
            
            for idx in group.indices:
                advantages[idx] = last_gae_lam
                returns[idx] = last_gae_lam + values[idx]
                
            # The current state's value becomes the next_value for the previous step
            next_value = values[group.indices[0]]

    batch = TransitionBatch(
        self_features=torch.from_numpy(np.asarray(self_rows, dtype=np.float32).reshape(-1, self_feature_dim())),
        candidate_features=torch.from_numpy(np.asarray(candidate_rows, dtype=np.float32).reshape(-1, empty_candidate[0], empty_candidate[1])),
        global_features=torch.from_numpy(np.asarray(global_rows, dtype=np.float32).reshape(-1, global_feature_dim())),
        candidate_mask=torch.from_numpy(np.asarray(candidate_masks, dtype=bool).reshape(-1, cfg.env.candidate_count)),
        target_index=torch.tensor(target_indices, dtype=torch.long),
        log_prob=torch.tensor(log_probs, dtype=torch.float32),
        returns=torch.tensor(returns, dtype=torch.float32),
        advantages=torch.tensor(advantages, dtype=torch.float32),
    )
    stats = {
        "episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "episodes_finished": float(len(episode_rewards)),
        "samples": float(len(values)),
    }
    return batch, batches, next_seed, stats