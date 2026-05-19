from typing import Protocol, Any, List
import torch
import numpy as np
import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

# Assuming encode_turn is available in your env file
from enviroment.utils import encode_turn

# ==========================================
# 1. The Standard Interface
# ==========================================
class BaseOpponent(Protocol):
    def act(self, raw_obs: Any) -> List[List[float]]:
        """Takes a raw Kaggle observation and returns a list of Kaggle moves."""
        ...

class FrozenPPOOpponent:
    def __init__(self, actor_network, config, obs_processor, act_processor):
        self.actor = actor_network
        self.config = config
        self.env_cfg = config.env
        self.obs_processor = obs_processor
        self.act_processor = act_processor
        
        # Ensure it is frozen upon initialization
        self.actor.eval()
        for param in self.actor.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def act(self, raw_obs) -> List[List[float]]:
        """Processes the observation and uses the frozen actor to pick moves."""
        batch = encode_turn(raw_obs, self.env_cfg)
        processed = self.obs_processor.process(batch, self.env_cfg)
        
        device = next(self.actor.parameters()).device
        
        self_feat = torch.from_numpy(processed["self_features"]).float().unsqueeze(0).to(device)
        cand_feat = torch.from_numpy(processed["candidates_features"]).float().unsqueeze(0).to(device)
        glob_feat = torch.from_numpy(processed["global_features"]).float().unsqueeze(0).to(device)
        mask = torch.from_numpy(processed["mask"]).bool().unsqueeze(0).to(device)
        
        logits = self.actor(self_feat, cand_feat, glob_feat, mask)
        action_indices = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
        action_indices = np.atleast_1d(action_indices).flatten()
        
        moves = self.act_processor.process(action_indices, batch.contexts, batch.state)
        return moves
    
    def sync_from(self, learner_actor):
        """Helper to cleanly sync weights from the main TorchRL learner during training."""
        learner_state = learner_actor.state_dict()
        # Clean the TorchRL wrapper prefixes automatically
        clean_state = {k.replace("module.", ""): v for k, v in learner_state.items()}
        self.actor.load_state_dict(clean_state)


class NearestNeighborOpponent:
    def __init__(self):
        pass
        
    def act(self, obs) -> List[List[float]]:
        moves = []
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        planets = [Planet(*p) for p in raw_planets]

        my_planets = [p for p in planets if p.owner == player]
        targets = [p for p in planets if p.owner != player]

        if not targets:
            return moves

        for mine in my_planets:
            nearest = min(targets, key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y))
            ships_needed = nearest.ships + 1
            if mine.ships >= ships_needed:
                angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
                moves.append([mine.id, angle, ships_needed])
        return moves
    
    def sync_from(self, learner_actor):
        pass