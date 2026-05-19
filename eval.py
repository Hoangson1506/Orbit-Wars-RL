import argparse
import importlib
import math
import random
import sys
import types
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import default_train_config_path, load_train_config
from agent import Agent, build_agent
from opponents import build_opponent

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, default=str(default_train_config_path()))
    parser.add_argument("--checkpoint", type=str, default="C:/Code/Orbit-Wars-RL/artifacts\orbit_wars_ppo/ckpt_last.pt")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--deterministic", action="store_true")
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

def extract_observation(state: Any) -> Any:
    if isinstance(state, dict):
        return state.get("observation")
    return getattr(state, "observation")


def extract_status(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("status", "UNKNOWN"))
    return str(getattr(state, "status", "UNKNOWN"))


def extract_reward(state: Any) -> float:
    if isinstance(state, dict):
        value = state.get("reward", 0.0)
    else:
        value = getattr(state, "reward", 0.0)
    return 0.0 if value is None else float(value)

def play_one_game(
    agent: Agent,
    opponent: Agent,
    *,
    seed: int,
) -> tuple[float, int]:
    from kaggle_environments import make

    env = make(
        "orbit_wars",
        configuration={"seed": int(seed), "randomSeed": int(seed)},
        debug=False,
    )
    env.reset(num_agents=2)
    states = env.step([[], []])
    player_obs = extract_observation(states[0])
    opponent_obs = extract_observation(states[1])
    done = extract_status(states[0]) != "ACTIVE"
    step_count = 0

    while not done:
        player_action = agent.act(player_obs)
        opponent_action = opponent.act(opponent_obs)
        states = env.step([player_action, opponent_action])
        player_obs = extract_observation(states[0])
        opponent_obs = extract_observation(states[1])
        done = extract_status(states[0]) != "ACTIVE"
        step_count += 1

    return extract_reward(states[0]), step_count

def reward_to_label(reward: float) -> str:
    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"
    return "draw"


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)
    device_name = args.device if args.device != "auto" else cfg.device
    device = resolve_device(device_name)
    seed_everything(args.seed)

    agent = build_agent(
        name="ppo",
        cfg=cfg,
        device=device,
        checkpoint_path=args.checkpoint,
        deterministic=args.deterministic
    )
    opponent = build_opponent(
        name="aggresive",
        cfg=cfg,
        device=device
    )

    wins = 0
    draws = 0
    losses = 0

    for game_idx in range(args.games):
        game_seed = args.seed + game_idx
        reward, steps = play_one_game(
            agent=agent,
            opponent=opponent,
            seed=game_seed,
        )
        label = reward_to_label(reward)
        if label == "win":
            wins += 1
        elif label == "loss":
            losses += 1
        else:
            draws += 1
        print(f"game={game_idx + 1} seed={game_seed} result={label} reward={reward:.1f} steps={steps}")

    total_games = max(args.games, 1)
    win_rate = wins / total_games
    print(f"summary wins={wins} losses={losses} draws={draws} games={args.games}")
    print(f"win_rate={win_rate:.4f}")


if __name__ == "__main__":
    main()


