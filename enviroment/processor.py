from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np
from typing import Any, List

from enviroment.utils import (
    TurnBatch,
    DecisionContext,
    self_feature_dim,
    candidate_feature_dim,
    global_feature_dim
)

class BaseObservationProcessor(ABC):
    @abstractmethod
    def get_space(self, env_cfg) -> gym.Space:
        """Returns the Gym Observation Space required for this strategy."""
        pass

    @abstractmethod
    def process(self, batch: TurnBatch, env_cfg) -> Any:
        """Converts a TurnBatch into the designated Gym Observation format."""
        pass

class BaseActionProcessor(ABC):
    @abstractmethod
    def get_space(self, env_cfg) -> gym.Space:
        """Returns the Gym Action Space required for this strategy."""
        pass

    @abstractmethod
    def process(self, gym_action: Any, contexts: List[DecisionContext], state) -> List[List[Any]]:
        """Converts a Gym action back into a list of Kaggle commands [[src, angle, ships], ...]."""
        pass


class PaddedObservationProcessor(BaseObservationProcessor):
    def get_space(self, env_cfg) -> gym.Space:
        return gym.spaces.Dict({
            "global": gym.spaces.Box(low=-1, high=1000, shape=(global_feature_dim(),), dtype=np.float32),
            "self": gym.spaces.Box(low=-1, high=1000, shape=(env_cfg.max_planets, self_feature_dim()), dtype=np.float32),
            "candidates": gym.spaces.Box(low=-1, high=1000, shape=(env_cfg.max_planets, env_cfg.candidate_count, candidate_feature_dim()), dtype=np.float32),
            "mask": gym.spaces.Box(low=0, high=1, shape=(env_cfg.max_planets, env_cfg.candidate_count), dtype=np.int8)
        })

    def process(self, batch: TurnBatch, env_cfg) -> dict:
        N = len(batch.contexts)
        pad_self = np.zeros((env_cfg.max_planets, self_feature_dim()), dtype=np.float32)
        pad_cand = np.zeros((env_cfg.max_planets, env_cfg.candidate_count, candidate_feature_dim()), dtype=np.float32)
        pad_mask = np.zeros((env_cfg.max_planets, env_cfg.candidate_count), dtype=np.int8)
        pad_global = batch.global_features[0] if N > 0 else np.zeros(global_feature_dim(), dtype=np.float32)

        if N > 0:
            limit = min(N, env_cfg.max_planets)
            pad_self[:limit] = batch.self_features[:limit]
            pad_cand[:limit] = batch.candidate_features[:limit]
            pad_mask[:limit] = batch.candidate_mask[:limit].astype(np.int8)

        return {"global": pad_global, "self": pad_self, "candidates": pad_cand, "mask": pad_mask}


class FixedActionProcessor(BaseActionProcessor):
    def get_space(self, env_cfg) -> gym.Space:
        # Array of integers: choose candidate index 0 (No Action) or index 1..N
        return gym.spaces.MultiDiscrete([env_cfg.candidate_count + 1] * env_cfg.max_planets)

    def process(self, gym_action: np.ndarray, contexts: list[DecisionContext], state) -> list:
        k_actions = []
        for act_val, ctx in zip(gym_action, contexts):
            if act_val == 0:
                continue
            cand_idx = act_val - 1
            if ctx.candidate_mask[cand_idx]:
                k_actions.append([ctx.source_id, ctx.target_angles[cand_idx], ctx.ship_counts[cand_idx]])
        return k_actions
    

class TransformerObservationProcessor(BaseObservationProcessor):
    def get_space(self, env_cfg) -> gym.Space:
        # Sequence processing requires varying sizes or sequential Dict spaces 
        return gym.spaces.Dict({
            "global_context": gym.spaces.Box(low=-1, high=1000, shape=(global_feature_dim(),), dtype=np.float32),
            "sequence_data": gym.spaces.Box(low=-1, high=1000, shape=(env_cfg.max_planets, self_feature_dim() + (env_cfg.candidate_count * candidate_feature_dim())), dtype=np.float32)
        })

    def process(self, batch: TurnBatch, env_cfg) -> dict:
        # Your custom logic to flatten tokens for an Attention layer without 
        # dense multidimensional matrix padding
        pass


class PointerNetworkActionProcessor(BaseActionProcessor):
    def get_space(self, env_cfg) -> gym.Space:
        # A pointer network wants a categorical target selection paired with a continuous scalar choice for custom ships
        return gym.spaces.Dict({
            "target_selection": gym.spaces.MultiDiscrete([env_cfg.candidate_count + 1] * env_cfg.max_planets),
            "ship_percentage_allocation": gym.spaces.Box(low=0.0, high=1.0, shape=(env_cfg.max_planets,), dtype=np.float32)
        })

    def process(self, gym_action: dict, contexts: list[DecisionContext], state) -> list:
        k_actions = []
        targets = gym_action["target_selection"]
        allocations = gym_action["ship_percentage_allocation"]

        for act_val, alloc_pct, ctx in zip(targets, allocations, contexts):
            if act_val == 0:
                continue
            cand_idx = act_val - 1
            if ctx.candidate_mask[cand_idx]:
                # --- FLEXIBLE SHIP MODIFICATION ---
                # Instead of running your fixed default logic (max(tgt.ships+1, 20)),
                # we look up the source planet state and let the policy scale its output fleet
                source_planet = next(p for p in state.planets if p.id == ctx.source_id)
                custom_ships = int(source_planet.ships * alloc_pct)
                
                if custom_ships > 0:
                    k_actions.append([ctx.source_id, ctx.target_angles[cand_idx], custom_ships])
        return k_actions