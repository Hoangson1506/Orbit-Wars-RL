from dataclasses import dataclass
from typing import Any

from config import TrainConfig
from features import TurnBatch, encode_turn
from opponents import OpponentPolicy
from game_types import GameState
from features import (
    TurnBatch, 
    parse_observation,
    encode_turn, 
)


@dataclass(slots=True)
class StepResult:
    batch: TurnBatch
    reward: float
    done: bool
    info: dict[str, Any]


class OrbitWarsEnv:
    def __init__(
        self,
        cfg: TrainConfig,
        opponent: OpponentPolicy,
        make_fn: Any | None = None,
        env_index: int = 0,
    ) -> None:
        self.cfg = cfg
        self.opponent = opponent
        self.make_fn = make_fn
        self.env_index = env_index
        self.env: Any | None = None
        self.last_obs: Any | None = None
        self.last_opp_obs: Any | None = None
        self.episode_index = 0
        self.learner_player = 0

    def reset(self, seed: int | None = None) -> TurnBatch:
        make_fn = self.make_fn or default_make_fn()
        configuration: dict[str, Any] = {}
        if seed is not None:
            configuration["seed"] = int(seed)
            configuration["randomSeed"] = int(seed)
        if self.cfg.alternate_player_sides:
            self.learner_player = (self.env_index + self.episode_index) % 2
        else:
            self.learner_player = 0
        self.env = make_fn("orbit_wars", configuration=configuration, debug=False)
        self.env.reset(num_agents=2)
        states = self.env.step([[], []])
        learner_state = states[self.learner_player]
        opponent_state = states[1 - self.learner_player]
        self.last_obs = extract_observation(learner_state)
        self.last_opp_obs = extract_observation(opponent_state)
        self.episode_index += 1
        return encode_turn(self.last_obs, self.cfg.env, env_index=self.env_index)
    
    def step(self, player_action: list[list[float | int]]) -> StepResult:
        if self.env is None:
            raise RuntimeError("Call reset() before step().")
        opponent_action = self.opponent.act(self.last_opp_obs)
        if self.learner_player == 0:
            joint_action = [player_action, opponent_action]
        else:
            joint_action = [opponent_action, player_action]
        states = self.env.step(joint_action)
        player_state = states[self.learner_player]
        opp_state = states[1 - self.learner_player]
        self.last_obs = extract_observation(player_state)
        self.last_opp_obs = extract_observation(opp_state)
        done = extract_status(player_state) != "ACTIVE"
        reward = terminal_reward(player_state, opp_state) if done else 0.0
        batch = encode_turn(self.last_obs, self.cfg.env, env_index=self.env_index)
        info = {
            "learner_player": self.learner_player,
            "player_status": extract_status(player_state),
            "opponent_status": extract_status(opp_state),
            "reward": reward,
        }
        return StepResult(batch=batch, reward=reward, done=done, info=info)
    
def default_make_fn() -> Any:
    from kaggle_environments import make

    return make


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

def terminal_reward(player_state: Any, opp_state: Any) -> float:
    player_reward = extract_reward(player_state)
    opponent_reward = extract_reward(opp_state)
    if player_reward > 0.0 and opponent_reward > 0.0:
        return 0.0
    return player_reward


# ==================================================
# REFACTOR FOR COORDINATED POLICY
# ==================================================
@dataclass(slots=True)
class StepResult:
    batch: TurnBatch
    reward: float
    done: bool
    info: dict[str, Any]

class OrbitWarsEnv:
    def __init__(
        self,
        cfg: TrainConfig,
        opponent: OpponentPolicy,
        make_fn: Any | None = None,
        env_index: int = 0,
    ) -> None:
        self.cfg = cfg
        self.opponent = opponent
        self.make_fn = make_fn
        self.env_index = env_index
        self.env: Any | None = None
        
        self.last_obs: Any | None = None
        self.last_opp_obs: Any | None = None
        self.last_state: GameState | None = None # Track state for dense rewards
        
        self.episode_index = 0
        self.learner_player = 0

    def reset(self, seed: int | None = None) -> TurnBatch:
        make_fn = self.make_fn or default_make_fn()
        configuration: dict[str, Any] = {}
        if seed is not None:
            configuration["seed"] = int(seed)
            configuration["randomSeed"] = int(seed)
            
        if self.cfg.alternate_player_sides:
            self.learner_player = (self.env_index + self.episode_index) % 2
        else:
            self.learner_player = 0
            
        self.env = make_fn("orbit_wars", configuration=configuration, debug=False)
        self.env.reset(num_agents=2)
        
        states = self.env.step([[], []])
        learner_state = states[self.learner_player]
        opponent_state = states[1 - self.learner_player]
        
        self.last_obs = extract_observation(learner_state)
        self.last_opp_obs = extract_observation(opponent_state)
        self.last_state = parse_observation(self.last_obs)
        
        self.episode_index += 1
        return encode_turn(self.last_obs, self.cfg.env, env_index=self.env_index)
    
    def step(self, player_action: list[list[float | int]]) -> StepResult:
        if self.env is None:
            raise RuntimeError("Call reset() before step().")
            
        opponent_action = self.opponent.act(self.last_opp_obs)
        
        if self.learner_player == 0:
            joint_action = [player_action, opponent_action]
        else:
            joint_action = [opponent_action, player_action]
            
        states = self.env.step(joint_action)
        player_state = states[self.learner_player]
        opp_state = states[1 - self.learner_player]
        
        self.last_obs = extract_observation(player_state)
        self.last_opp_obs = extract_observation(opp_state)
        current_state = parse_observation(self.last_obs)
        
        done = extract_status(player_state) != "ACTIVE"
        
        # --- Reward Calculation ---
        term_reward = terminal_reward(player_state, opp_state) if done else 0.0
        dense_reward = 0.0
        
        if not done and self.last_state is not None:
            dense_reward = calculate_dense_reward(self.last_state, current_state, self.learner_player)
            
        total_reward = term_reward + dense_reward
        self.last_state = current_state
        
        batch = encode_turn(self.last_obs, self.cfg.env, env_index=self.env_index)
        
        info = {
            "learner_player": self.learner_player,
            "player_status": extract_status(player_state),
            "opponent_status": extract_status(opp_state),
            "reward": total_reward, # Log total for PPO
            "terminal_reward": term_reward, # Log terminal separately for metrics
            "dense_reward": dense_reward,
        }
        
        return StepResult(batch=batch, reward=total_reward, done=done, info=info)


# --- Reward Shaping Logic ---
def calculate_dense_reward(old_state: GameState, new_state: GameState, player_id: int) -> float:
    """Provides small incremental rewards to speed up PPO convergence."""
    
    def count_metrics(state: GameState):
        planets_owned = 0
        total_ships = 0
        for p in state.planets:
            if p.owner == player_id:
                planets_owned += 1
                total_ships += p.ships
        for f in state.fleets:
            if f.owner == player_id:
                total_ships += f.ships
        return planets_owned, total_ships

    old_planets, old_ships = count_metrics(old_state)
    new_planets, new_ships = count_metrics(new_state)
    
    delta_planets = new_planets - old_planets
    delta_ships = new_ships - old_ships
    
    # Weights should be small enough not to overshadow the terminal reward (1.0 or -1.0)
    # E.g., gaining a planet is worth 0.05, losing one is -0.05
    return (delta_planets * 0.05) + (delta_ships * 0.0005)

def default_make_fn() -> Any:
    from kaggle_environments import make
    return make

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

def terminal_reward(player_state: Any, opp_state: Any) -> float:
    player_reward = extract_reward(player_state)
    opponent_reward = extract_reward(opp_state)
    if player_reward > 0.0 and opponent_reward > 0.0:
        return 0.0
    return player_reward