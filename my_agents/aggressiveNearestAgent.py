from my_agents.baseAgent import BaseAgent
from my_agents.utils import compute_move_angle
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
from utils.registry import register_model
import math
import numpy as np

@register_model("AggressiveNearestAgent")
class AggressiveNearestAgent(BaseAgent):
    def __init__(self, static_max_distance=40, dynamic_max_distance=30, min_ships=5, max_turns=20, max_neutral=35, **kwargs):
        super().__init__()
        self.awaiting_results = {}
        self.dynamic_max_distance=dynamic_max_distance
        self.static_max_distance=static_max_distance
        self.min_ships = min_ships
        self.max_turns = max_turns
        self.max_neutral = max_neutral
        self.orbiting_planets = None
        self.angular_velocity = None

    def forward(self, x):
        return x

    def act(self, obs, config=None, **kwargs):
        moves = []
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        planets = [Planet(*p) for p in raw_planets]
        comets = obs.comet_planet_ids

        owned_planets = []
        target_planets = []
        for p in planets:
            if p.owner == player:
                owned_planets.append(p)
            elif p.id not in comets:
                if p.owner == -1 and p.ships > self.max_neutral:
                    continue
                else:
                    target_planets.append(p)

        owned_indexes = {p.id: i for i, p in enumerate(owned_planets)}
        target_indexes = {p.id: i for i, p in enumerate(target_planets)}

        #initialize orbiting planets and angular velocity
        if self.orbiting_planets is None:
            self.orbiting_planets = {}
            for p in planets:
                orbit_r = math.hypot(p.x - 50, p.y - 50)
                if orbit_r + p.radius < 50:
                    self.orbiting_planets[p.id] = orbit_r
            self.angular_velocity = obs.angular_velocity

        if len(target_planets) == 0:
            return moves
        
        # timer
        for tid, (turns, owner) in self.awaiting_results.items():
            if turns > 0:
                self.awaiting_results[tid][0] -= 1

        # if a planet owner changes, reset awaiting results for that planet
        for planet in planets:
            if planet.id in self.awaiting_results:
                if self.awaiting_results[planet.id][1] != planet.owner:
                    self.awaiting_results[planet.id] = [0, planet.owner]
            
        # after the early game, there will be planets that have no target in range
        # they will move all their ships to owned planets in range that has targets in range
        owned_matrix = np.zeros((len(owned_planets), len(owned_planets)))
        distance_matrix = np.zeros((len(owned_planets), len(target_planets)))

        for i, op1 in enumerate(owned_planets):
            for j, op2 in enumerate(owned_planets):
                owned_matrix[i][j] = math.hypot(op1.x - op2.x, op1.y - op2.y)
            for k, tp in enumerate(target_planets):
                distance_matrix[i][k] = math.hypot(op1.x - tp.x, op1.y - tp.y)

        # TODO: The agent only checks for the number of ships on the owned and target planet,
        #       it should also check for incoming fleets to the target planet, 
        #       both from the player and the opponent, to make a more informed decision.
        #       For now, to avoid sending multiple fleets to the same target, 
        #       we will approximate turn needed and reduce by 1 each turn


        # TODO: Comets are also not checked
        #       for now, ignore them completely
        sorted_targets = sorted(target_planets, key=lambda p: p.ships)

        ships_available = {op.id: op.ships for op in owned_planets}

        #try to improve early game optimization, maybe double loop?
        # instead of getting the most production, least time is better
        # or instead of looping through targets, loop through owned and find all their target in range
        # we can use that to coordinate attacks
        # also if static planet has target in range but all targets are on cooldown, send ships to ally
        # check all ships on board
        #also if a neutral planet's number of ships is too high (> 40ish), don't attack
        # take neutral planets from min ship of max ship
        for tp in sorted_targets:
            if tp.id in self.awaiting_results and self.awaiting_results[tp.id][0] > 0:
                continue

            j = target_indexes[tp.id]
            if tp.id in self.orbiting_planets:
                max_distance = self.dynamic_max_distance
            else:
                max_distance = self.static_max_distance
            
            available = sum([ships_available[op.id]
                             for op in owned_planets
                             if distance_matrix[owned_indexes[op.id]][j] <= max_distance])
            
            # a player's planet will produce ships every turn
            if tp.owner != -1:
                if tp.id not in self.orbiting_planets:
                    extra_ships = 10 * tp.production
                else:
                    extra_ships = 20 * tp.production
            else:
                extra_ships = 0

            ships_needed = tp.ships + extra_ships + 1
            if available <= ships_needed:
                continue

            candidate_moves = []
            wait_time = 0
            sorted_owned = sorted(owned_planets, key=lambda p: distance_matrix[owned_indexes[p.id]][j])
            for op in sorted_owned:
                i = owned_indexes[op.id]
                if distance_matrix[i][j] > max_distance:
                    break

                if ships_available[op.id] <= 0:
                    continue

                # only send if total ship of each enemy player in range is less than ours
                # the center is our current planet, as we don't want to lose it
                if op.id in self.orbiting_planets:
                    md = self.dynamic_max_distance
                else:
                    md = self.static_max_distance
                total_allies_in_range = sum([ships_available[op2.id]
                                            for op2 in owned_planets
                                            if owned_matrix[owned_indexes[op2.id]][i] <= md])
                enemies = {-1: 0}
                for tp2 in target_planets:
                    if tp2.owner == -1:
                        continue
                    else:
                        tp2_idx = target_indexes[tp2.id]

                        if distance_matrix[i][tp2_idx] <= md:
                            if tp2.owner not in enemies:
                                enemies[tp2.owner] = 0
                            enemies[tp2.owner] += tp2.ships

                if max(enemies.values()) > total_allies_in_range:
                    continue

                send = min(ships_available[op.id], ships_needed)
                if send == 0:
                    continue
                elif send < self.min_ships:
                    send = min(self.min_ships, ships_available[op.id])

                result = compute_move_angle(op, tp, send, planets, self.orbiting_planets, comets, self.angular_velocity, self.max_turns)
                if result is not None:
                    candidate_moves.append((op.id, result[0], send))
                    wait_time = max(wait_time, result[1])
                    ships_available[op.id] -= send
                    ships_needed -= send

                if ships_needed <= 0:
                    moves += candidate_moves
                    self.awaiting_results[tp.id] = [wait_time, tp.owner]
                    break

            if ships_needed > 0:
                # if we cannot send enough ships, revert the candidate moves
                for op_id, _, send in candidate_moves:
                    ships_available[op_id] += send

        # search for neighbor in range
        # when considering reinforce, check for incoming enemy fleets to avoid losing a planet
        orbiting_target_indexes = [i for id, i in target_indexes.items() if id in self.orbiting_planets]
        static_target_indexes = [i for id, i in target_indexes.items() if id not in self.orbiting_planets]
        for i, op in enumerate(owned_planets):
            if op.id in self.orbiting_planets:
                continue 
            # check if there is any target in range
            static_target_in_range = np.any(distance_matrix[i, static_target_indexes] <= self.static_max_distance)
            orbiting_target_in_range = np.any(distance_matrix[i, orbiting_target_indexes] <= self.dynamic_max_distance)
            if static_target_in_range or orbiting_target_in_range:
                continue
            sorted_indices = np.argsort(owned_matrix[i])
            # 0 is itself, so start from 1
            candidates = []
            for j in sorted_indices[1:]:
                if owned_planets[j].id not in self.orbiting_planets:
                    max_distance = self.static_max_distance
                else:
                    max_distance = self.dynamic_max_distance
                if owned_matrix[i][j] <= max_distance:
                    if np.any(distance_matrix[j, static_target_indexes] <= self.static_max_distance):
                        candidates.append(j)
                    elif np.any(distance_matrix[j, orbiting_target_indexes] <= self.dynamic_max_distance):
                        candidates.append(j)
                else:
                    break
            if candidates:
                send = math.floor((ships_available[op.id] - self.max_neutral) / len(candidates))
                if send >= self.min_ships:
                    for j in candidates:
                        result = compute_move_angle(op, owned_planets[j], send, planets, self.orbiting_planets, comets, self.angular_velocity, self.max_turns)
                        if result is not None:
                            ships_available[op.id]-= send
                            moves.append((op.id, result[0], send))
            else:
                # try to send to 2 nearset owned orbiting planets as they are the attacker
                send = math.floor((ships_available[op.id] - self.max_neutral) / 2)
                if send >= self.min_ships:
                    count = 0
                    for j in sorted_indices[1:]:
                        if count > 1:
                            break
                        if owned_planets[j].id in self.orbiting_planets:
                            result = compute_move_angle(op, owned_planets[j], send, planets, self.orbiting_planets, comets, self.angular_velocity, self.max_turns)
                            if result is not None:
                                count +=1
                                ships_available[op.id] -= send
                                moves.append((op.id, result[0], send))

        return moves