from __future__ import annotations

import math
from typing import Any

try:
    import gymnasium as gym
except ImportError:  # Keep action translation importable without Gymnasium installed.
    gym = None

from datasets import (
    OrbitWarsGraphBuilder,
    _obs_get,
    _planet_id,
    _planet_ships,
    _planet_x,
    _planet_y,
    _player_id,
    _sorted_planets,
)


BaseEnv = gym.Env if gym is not None else object


class OrbitWarsPyGWrapper(BaseEnv):
    """
    Thin Gym wrapper around an Orbit Wars environment.

    The wrapper does two things:
      1. converts raw observations into PyG graphs
      2. converts a graph action `(source_idx, target_idx, ship_pct)` into one
         Orbit Wars move `[source_planet_id, angle, ship_count]`
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        kaggle_env: Any,
        *,
        agent_id: int = 0,
        graph_builder: OrbitWarsGraphBuilder | None = None,
    ) -> None:
        super().__init__()
        self.kaggle_env = kaggle_env
        self.agent_id = agent_id
        self.graph_builder = graph_builder or OrbitWarsGraphBuilder()
        self.current_raw_obs: Any | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if gym is not None:
            super().reset(seed=seed)
        raw_reset = self.kaggle_env.reset()
        self.current_raw_obs = _agent_value(raw_reset, self.agent_id)
        return self._graph(), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        moves = self.action_to_moves(action)
        raw_obs, reward, done, info = self.kaggle_env.step(moves)
        self.current_raw_obs = _agent_value(raw_obs, self.agent_id)
        reward_value = _agent_value(reward, self.agent_id)
        done_value = _agent_value(done, self.agent_id)
        return (
            self._graph(),
            0.0 if reward_value is None else float(reward_value),
            bool(done_value),
            False,
            info,
        )

    def action_to_moves(self, action: Any) -> list[list[int | float]]:
        if self.current_raw_obs is None:
            raise RuntimeError("Call reset() before translating actions.")

        source_idx, target_idx, ship_pct = _unpack_action(action)
        move = graph_action_to_move(
            self.current_raw_obs,
            source_idx=source_idx,
            target_idx=target_idx,
            ship_pct=ship_pct,
        )
        return [] if move is None else [move]

    def _graph(self) -> Any:
        if self.current_raw_obs is None:
            raise RuntimeError("No observation is available.")
        return self.graph_builder.obs_to_data(
            self.current_raw_obs,
            _player_id(self.current_raw_obs, self.agent_id),
        )

def _unpack_action(action: Any) -> tuple[int, int, float]:
    if isinstance(action, dict):
        return (
            _to_int(action["source"]),
            _to_int(action["target"]),
            _to_float(action["ship_pct"]),
        )

    source_idx, target_idx, ship_pct = action
    return _to_int(source_idx), _to_int(target_idx), _to_float(ship_pct)


def _agent_value(value: Any, agent_id: int) -> Any:
    if isinstance(value, tuple) and len(value) == 2:
        value = value[0]
    if isinstance(value, list) and len(value) > agent_id:
        value = value[agent_id]
    if isinstance(value, dict) and "observation" in value:
        return value["observation"]
    return _obs_get(value, "observation", value)


def _to_int(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _to_float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)
