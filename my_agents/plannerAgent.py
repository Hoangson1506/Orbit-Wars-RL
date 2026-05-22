from my_agents.baseAgent import BaseAgent
from my_agents.utils import static_collision_check, orbiting_collision_check, compute_move_angle
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
from utils.registry import register_model
import math
import numpy as np

# --- SELF NOTE ---
# The distance between orbiting and oribiting is a constant 
# (rotating at same speed around one point)
# same for static and static

# This means at any given turn, the time needed for the same fleet to go from one planet
# to another of the same kind is unchanged

# Therefore, we can set up a priority queue that consider the turns needed to capture a planet
# It will be updated every turn, with the only changes being number of ships and 
# static-to-orbiting relation

# This is to avoid instead of waiting a few turns to take a nearby planet,
# the agent immediately go for a much further one, which takes much more time to capture


@register_model("PlannerAgent")
class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__(self, max_neutral=40, max_look_ahead=20)
        self.checked_fleets = {} # fleet id : [owner, target id, ships, remaining turns until arrival]
        self.orbiting_planets = None # list of orbiting planets's id
        self.angular_velociry = None # The current game's angular velocity
        self.max_neutral = self.max_neutral # don't attack neutral planets with too many ships
        self.max_look_ahead = self.max_look_ahead
        # the result after all fleets heading toward it arrives
        self.expected_state = {} # planet id : [s1, s2, ...], s = (owner id, ships)
        self.sun = Planet(-1, -1, 50, 50, 10, 0, 0)

    def forward(self, x):
        return super().forward(x)
    
    def act(self, obs):
        moves = []
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        planets = [Planet(*p) for p in raw_planets]
        raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
        fleets = [Fleet(*f) for f in raw_fleets]
        comets = obs.comet_planet_ids

        owned_planets = []
        target_planets = []
        for p in planets:
            if p.id in comets:
                continue
            elif p.owner == player:
                owned_planets.append(p)
            else:
                if p.owner == -1 and p.ships > self.max_neutral:
                    continue
                else:
                    target_planets.append(p)

        owned_indexes = {p.id: i for i, p in enumerate(owned_planets)}
        target_indexes = {p.id: i for i, p in enumerate(target_planets)}

        # initialize orbiting planets, angular velocity and expected state keys
        if self.orbiting_planets is None:
            self.angular_velocity = obs.angular_velocity
            self.orbiting_planets = {}
            for p in planets:
                orbit_r = math.hypot(p.x - 50, p.y - 50)
                if orbit_r + p.radius < 50:
                    self.orbiting_planets[p.id] = orbit_r
            

        if len(target_planets) == 0:
            return moves
        
        #
        
        # check for new fleets and update the expected state
        for f in fleets:
            if f.id in self.checked_fleets and self.checked_fleets[f.id] is not None:
                self.checked_fleets[f.id][3] -= 1
                if self.checked_fleets[f.id][3] <= 0:
                    self.checked_fleets[f.id] = None
            else:
                fleet_speed = 1.0 + 5.0 * (math.log(f.ships) / math.log(1000)) ** 1.5
                remaining_turns = 501
                target = None

                # check where it's going
                dynamic_result =  orbiting_collision_check(f.x, f.y, f.angle, fleet_speed, f.from_planet_id, planets, self.orbiting_planets, comets, self.angular_velocity, self.max_look_ahead)
                if dynamic_result is not None:
                    if dynamic_result[1] < remaining_turns:
                        target = dynamic_result[0]
                        remaining_turns = dynamic_result[1]
                static_result = static_collision_check(f.x, f.y, f.angle, fleet_speed, f.from_planet_id, planets + [self.sun], self.orbiting_planets)
                if static_result is not None:
                    if static_result[0] == -1 and static_result[1] < remaining_turns: # hit the sun first
                        target = None
                    if static_result[1] < remaining_turns:
                        target = static_result[0]
                        remaining_turns = static_result[1]

                if target is not None:
                    self.checked_fleets[f.id] = [f.owner, target, f.ships, remaining_turns]

                    # update the expected state, starting from the turn the fleet arrives
                    turn_index = math.ceil(remaining_turns) - 1
