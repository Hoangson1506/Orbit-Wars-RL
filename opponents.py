from typing import Any, Protocol
import math

import torch
import numpy as np
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from config import TrainConfig
from features import encode_turn
from agent import build_policy
from policy import PlanetPolicy
from algorithms.ppo import sample_actions


class OpponentPolicy(Protocol):
    def act(self, observation: Any) -> list[list[float | int]]:
        ...


class KaggleRandomOpponent:
    def __init__(self) -> None:
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent

        self._agent = random_agent

    def act(self, observation: Any) -> list[list[float | int]]:
        payload = {
            "player": obs_get(observation, "player", 0),
            "planets": list(obs_get(observation, "planets", [])),
        }
        return list(self._agent(payload))
    

class NearestPlanetOpponent:
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

    
class AggressiveNearestOpponent:
    def __init__(self, max_distance=30, min_ships=5, **kwargs):
        super().__init__()
        self.awaiting_results = {}
        self.max_distance = max_distance
        self.min_ships = min_ships

    def forward(self, x):
        return x

    def compute_move_angle(self, op, tp, send, angular_velocity):
        dx = tp.x - op.x
        dy = tp.y - op.y

        orbit_r = math.hypot(tp.x - 50, tp.y - 50)
        is_orbiting = orbit_r + tp.radius < 50
        if not is_orbiting:
            angle = math.atan2(dy, dx)
            distance = math.hypot(dx, dy)
            fleet_speed = 1.0 + 5.0 * (math.log(send) / math.log(1000)) ** 1.5
            turns_needed = math.ceil(distance / fleet_speed)

            if tp.id in self.awaiting_results:
                self.awaiting_results[tp.id][0] = max(self.awaiting_results[tp.id][0], turns_needed)
            else:
                self.awaiting_results[tp.id] = [turns_needed, tp.owner]

            return angle

        else:
            fleet_speed = 1.0 + 5.0 * (math.log(send) / math.log(1000)) ** 1.5
            phi = math.atan2(tp.y - 50, tp.x - 50)

            for t in range(1, self.max_distance * 2):
                new_tp_x = 50 + orbit_r * math.cos(phi + angular_velocity * t)
                new_tp_y = 50 + orbit_r * math.sin(phi + angular_velocity * t)

                angle = math.atan2(new_tp_y - op.y, new_tp_x - op.x)

                fleet_starting_x = op.x + op.radius * math.cos(angle)
                fleet_starting_y = op.y + op.radius * math.sin(angle)
                new_fleet_x = fleet_starting_x + fleet_speed * t * math.cos(angle)
                new_fleet_y = fleet_starting_y + fleet_speed * t * math.sin(angle)

                # if at one point the fleet is in the sun, abort the move
                if math.hypot(new_fleet_x - 50, new_fleet_y - 50) < 10:
                    return None

                if math.hypot(new_fleet_x - new_tp_x, new_fleet_y - new_tp_y) < tp.radius:
                    if tp.id in self.awaiting_results:
                        self.awaiting_results[tp.id][0] = max(self.awaiting_results[tp.id][0], t)
                    else:
                        self.awaiting_results[tp.id] = [t, tp.owner]
                    return angle

        return None


    def act(self, obs, config=None, **kwargs):
        moves = []
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        planets = [Planet(*p) for p in raw_planets]
        raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
        fleets = [Fleet(*f) for f in raw_fleets]

        owned_planets = [p for p in planets if p.owner == player]
        target_planets = [p for p in planets if p.owner != player]

        if not target_planets:
            return moves
        
        # timer
        for tid, (turns, owner) in self.awaiting_results.items():
            if turns > 0:
                self.awaiting_results[tid][0] -= 1

        # if a planet owner changed and it's not us, reset awaiting results for that planet
        for planet in planets:
            if planet.id in self.awaiting_results:
                if self.awaiting_results[planet.id][1] != planet.owner and planet.owner != player:
                    del self.awaiting_results[planet.id]


        distance_matrix = np.zeros((len(owned_planets), len(target_planets)))
        for i, op in enumerate(owned_planets):
            for j, tp in enumerate(target_planets):
                distance_matrix[i][j] = math.hypot(op.x - tp.x, op.y - tp.y)

        # after the early game, there will be planets that have no target in range
        # they will move all their ships to owned planets in range that has targets in range
        owned_matrix = np.zeros((len(owned_planets), len(owned_planets)))
        for i, op1 in enumerate(owned_planets):
            for j in range(i + 1, len(owned_planets)):
                op2 = owned_planets[j]
                owned_matrix[i][j] = math.hypot(op1.x - op2.x, op1.y - op2.y)
                owned_matrix[j][i] = owned_matrix[i][j]

        # search for neighbor in range
        for i, op in enumerate(owned_planets):
            # check if there is any target in range
            target_in_range = np.any(distance_matrix[i] <= self.max_distance)
            if target_in_range:
                continue
            sorted_indices = np.argsort(owned_matrix[i])
            # 0 is itself, so start from 1
            candidates = []
            for j in sorted_indices[1:]:
                if owned_matrix[i][j] <= self.max_distance:
                    if np.any(distance_matrix[j] <= self.max_distance):
                        candidates.append(j)
                else:
                    break
            if candidates:
                send = math.floor(op.ships / len(candidates))
                if send >= self.min_ships:
                    for j in candidates:
                        angle = self.compute_move_angle(op, owned_planets[j], send, obs.angular_velocity)
                        if angle is not None:
                            moves.append((op.id, angle, send))
            else:
                # just split and send to 3 nearest planets
                send = math.floor(op.ships / 3)
                if send >= self.min_ships:
                    for j in sorted_indices[1:4]:
                        angle = self.compute_move_angle(op, owned_planets[j], send, obs.angular_velocity)
                        if angle is not None:
                            moves.append((op.id, angle, send))

        # TODO: The agent only checks for the number of ships on the owned and target planet,
        #       it should also check for incoming fleets to the target planet, 
        #       both from the player and the opponent, to make a more informed decision.
        #       For now, to avoid sending multiple fleets to the same target, 
        #       we will approximate turn needed and reduce by 1 each turn


        # TODO: Comets are also not checked
        #       for now, ignore them completely
        comet_ids = obs.comet_planet_ids

        sorted_targets = sorted(target_planets, key=lambda p: p.production, reverse=True)

        ships_available = {planet.id: planet.ships for planet in planets}

        for tp in sorted_targets:
            if tp.id in comet_ids:
                continue

            if tp.id in self.awaiting_results and self.awaiting_results[tp.id][0] > 0:
                continue

            j = target_planets.index(tp)
            nearest_idx = np.argsort(distance_matrix[:, j])[:3]
            available = sum([ships_available[owned_planets[i].id] for i in nearest_idx 
                             if distance_matrix[i][j] <= self.max_distance])
            
            # a player's planet will produce ships every turn
            if tp.owner != -1:
                extra_ships = 8 * tp.production

            else:
                extra_ships = 0

            ships_needed = tp.ships + extra_ships + 1

            if available <= ships_needed:
                continue
            
            for i in nearest_idx:
                op = owned_planets[i]
                if ships_available[op.id] <= 0:
                    continue

                send = min(ships_available[op.id], ships_needed)
                if send == 0:
                    continue
                elif send < self.min_ships:
                    send = min(self.min_ships, ships_available[op.id])
                
                ships_available[op.id] -= send
                ships_needed -= send

                angle = self.compute_move_angle(op, tp, send, obs.angular_velocity)
                if angle is not None:
                    moves.append((op.id, angle, send))
                else:
                    ships_available[op.id] += send
                    ships_needed += send

                if ships_needed <= 0:
                    break

        return moves
    
class SelfPlayOpponent:
    def __init__(self, cfg: TrainConfig, device: torch.device, deterministic: bool = True) -> None:
        from features import candidate_feature_dim, global_feature_dim, self_feature_dim

        self.cfg = cfg
        self.device = device
        self.deterministic = deterministic
        self.policy = build_policy(cfg=cfg, device=device)
        self.policy.eval()

    def sync_from(self, source_policy: PlanetPolicy) -> None:
        state_dict = source_policy.state_dict()
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

    def act(self, observation: Any) -> list[list[float | int]]:
        batch = encode_turn(observation, self.cfg.env, env_index=0)
        if batch.self_features.shape[0] == 0:
            return []
        with torch.inference_mode():
            outputs = self.policy(
                torch.from_numpy(batch.self_features).to(self.device),
                torch.from_numpy(batch.candidate_features).to(self.device),
                torch.from_numpy(batch.global_features).to(self.device),
                torch.from_numpy(batch.candidate_mask).to(self.device).bool(),
            )
            sampled = sample_actions(outputs, deterministic=self.deterministic)
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


def build_opponent(
    name: str,
    cfg: TrainConfig | None = None,
    device: torch.device | None = None,
) -> OpponentPolicy:
    if name == "random":
        return KaggleRandomOpponent()
    if name == "aggresive":
        return AggressiveNearestOpponent()
    if name == "nearest":
        return NearestPlanetOpponent()
    if name == "self":
        if cfg is None or device is None:
            raise ValueError("cfg and device are required for self opponent")
        return SelfPlayOpponent(cfg, device=device, deterministic=cfg.self_play_deterministic)
    raise ValueError(f"Unknown opponent: {name}")


def obs_get(observation: Any, key: str, default: Any) -> Any:
    if isinstance(observation, dict):
        return observation.get(key, default)
    return getattr(observation, key, default)