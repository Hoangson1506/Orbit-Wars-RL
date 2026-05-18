from abc import ABC, abstractmethod
import torch    
import torch.nn as nn
import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from utils.registry import register_model

class BaseAgent(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def act(self, obs):
        pass

@register_model("NearestPlanetAgent")
class NearestPlanetAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

    def act(self, obs, config=None, **kwargs):
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