from typing import Any, Protocol
import math
import sys
import types
import importlib

import torch
import numpy as np
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from config import TrainConfig
from features import encode_turn, decode_network_actions
from policy import PlanetPolicy, TransformerPlanetPolicy
from algorithms.ppo import sample_actions
from features import TurnBatch, self_feature_dim, candidate_feature_dim, global_feature_dim, planet_feature_dim


class Agent(Protocol):
    def act(self, observation: Any) -> list[list[float | int]]:
        ...


# ===============================================
# PPO Agent
# ===============================================
def build_policy(cfg: TrainConfig, device: torch.device) -> PlanetPolicy:
    arch = cfg.model.architecture.lower()

    if arch == "mlp":
        return PlanetPolicy(
            self_dim=self_feature_dim(),
            candidate_dim=candidate_feature_dim(),
            global_dim=global_feature_dim(),
            candidate_count=cfg.env.candidate_count,
            hidden_size=cfg.model.hidden_size,
        ).to(device)
    if arch == "transformer":
        return TransformerPlanetPolicy(
            self_dim=self_feature_dim(),
            candidate_dim=candidate_feature_dim(),
            global_dim=global_feature_dim(),
            candidate_count=cfg.env.candidate_count,
            hidden_size=cfg.model.hidden_size,
            num_heads=cfg.model.num_heads, # Passed specific to transformer
        ).to(device)
    if arch == "coordinated":
        return TransformerPlanetPolicy(
            planet_dim=planet_feature_dim(),
            global_dim=global_feature_dim(),
            hidden_size=cfg.model.hidden_size,
            num_heads=cfg.model.num_heads, # Passed specific to transformer
            num_layers=cfg.model.num_layers
        ).to(device)
    
    raise ValueError(f"Unknown model architecture: {arch}")

def register_checkpoint_module_aliases() -> None:
    sys.modules.setdefault("src", types.ModuleType("src"))
    sys.modules.setdefault("src.rl_template", types.ModuleType("src.rl_template"))
    module_candidates = {
        "config": ["src.rl_template.config", "src.config", "config"],
        "features": ["src.rl_template.features", "src.features", "features"],
        "policy": ["src.rl_template.policy", "src.policy", "policy"],
        "ppo": ["src.rl_template.ppo", "src.ppo", "ppo"],
        "game_types": ["src.rl_template.game_types", "src.game_types", "game_types"],
        "opponents": ["src.rl_template.opponents", "src.opponents", "opponents"],
        "env": ["src.rl_template.env", "src.env", "env"],
        "train": ["src.rl_template.train", "src.train", "train"],
    }

    for canonical_name, candidates in module_candidates.items():
        module = None
        for candidate in candidates:
            try:
                module = importlib.import_module(candidate)
                break
            except ModuleNotFoundError:
                continue
        if module is None:
            continue
        sys.modules[f"src.rl_template.{canonical_name}"] = module
        sys.modules[f"src.{canonical_name}"] = module

def load_checkpoint_if_available(policy: PlanetPolicy, checkpoint_path: str | None, device: torch.device) -> None:
    register_checkpoint_module_aliases()
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("policy", checkpoint)
    policy.load_state_dict(state_dict)


class PPOAgent:
    """Wraps a neural network policy to conform to the Agent interface."""
    def __init__(
        self, 
        policy: PlanetPolicy, 
        cfg: TrainConfig, 
        device: torch.device, 
        deterministic: bool
    ):
        self.policy = policy
        self.cfg = cfg
        self.device = device
        self.deterministic = deterministic

    def act(self, obs: Any) -> list[list[float | int]]:
        # PPO-specific observation processing
        batch = encode_turn(obs, self.cfg.env, env_index=0)
        return self._build_moves(batch, self.policy, self.device, self.deterministic)
    
    def _build_moves(self, batch: TurnBatch, policy: PlanetPolicy, device: torch.device, deterministic: bool) -> list[list[float | int]]:
        if batch.self_features.shape[0] == 0:
            return []
        with torch.inference_mode():
            outputs = policy(
                torch.from_numpy(batch.self_features).to(device),
                torch.from_numpy(batch.candidate_features).to(device),
                torch.from_numpy(batch.global_features).to(device),
                torch.from_numpy(batch.candidate_mask).to(device).bool(),
            )
            sampled = sample_actions(outputs, deterministic=deterministic)
        target_indices = sampled.target_index.detach().cpu().numpy()
        moves: list[list[float | int]] = []
        for row_idx, context in enumerate(batch.contexts):
            target_idx = int(target_indices[row_idx])
            if target_idx == 0:
                continue
            if target_idx >= len(context.candidate_ids):
                continue
            if not context.candidate_mask[target_idx]:
                continue
            ships = int(context.ship_counts[target_idx])
            if ships <= 0:
                continue
            moves.append([context.source_id, float(context.target_angles[target_idx]), ships])
        return moves
    

def build_agent(
    name: str,
    cfg: TrainConfig | None = None,
    device: torch.device | None = None,
    checkpoint_path: str | None = None,
    deterministic: bool = True
) -> Agent:
    if name == "ppo":
        if cfg is None or device is None:
            raise ValueError("cfg and device are required for self opponent")
        policy = build_policy(cfg=cfg, device=device)
        load_checkpoint_if_available(policy, checkpoint_path, device)
        policy.eval()
        return PPOAgent(
            policy=policy, 
            cfg=cfg, 
            device=device, 
            deterministic=deterministic
        )
    raise ValueError(f"Unknown opponent: {name}")

class PPOAgent:
    """Wraps a neural network policy to conform to the Agent interface."""
    def __init__(
        self, 
        policy: PlanetPolicy, 
        cfg: TrainConfig, 
        device: torch.device, 
        deterministic: bool
    ):
        self.policy = policy
        self.cfg = cfg
        self.device = device
        self.deterministic = deterministic
        self.policy.eval() # Ensure the policy is always in eval mode during inference

    def act(self, obs: Any) -> list[list[float | int]]:
        # 1. Encode the turn into the new sequence format
        batch = encode_turn(obs, self.cfg.env, env_index=0)
        return self._build_moves(batch)
    
    def _build_moves(self, batch: TurnBatch) -> list[list[float | int]]:
        # 2. Add batch dimension (B=1) and move to device
        planet_feat = torch.from_numpy(batch.planet_features).unsqueeze(0).to(self.device)
        global_feat = torch.from_numpy(batch.global_features).unsqueeze(0).to(self.device)
        target_mask = torch.from_numpy(batch.target_mask).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            # 3. Forward pass through the Transformer
            outputs = self.policy(planet_feat, global_feat)

            # 4. Sample sequence actions
            sampled = sample_actions(
                outputs, 
                target_mask=target_mask, 
                deterministic=self.deterministic
            )

        # 5. Extract the first (and only) item in the batch
        target_indices = sampled.target_index[0].cpu().tolist()
        actor_mask = batch.actor_mask.tolist()

        # 6. Decode back into Kaggle moves using the shared physics helper
        moves = decode_network_actions(target_indices, actor_mask, batch)

        return moves