import functools

from pettingzoo import ParallelEnv
from kaggle_environments import make

from enviroment.processor import BaseActionProcessor, BaseObservationProcessor
from enviroment.utils import encode_turn

class OrbitWarsWrapper(ParallelEnv):
    metadata = {
        "name": "orbit_wars_v0"
    }
    def __init__(self, config, obs_processor: BaseObservationProcessor, act_processor: BaseActionProcessor):
        super().__init__()
        self.config = config
        self.env_cfg = config.env
        self.obs_processor = obs_processor
        self.act_processor = act_processor

        # Agents
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]

        # Dynamically set spaces from strategies
        self.observation_spaces = {
            agent: obs_processor.get_space(self.env_cfg)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: act_processor.get_space(self.env_cfg)
            for agent in self.possible_agents
        }

        self.k_env = make("orbit_wars", debug=False)
        self.trainer = None
        self.current_contexts = {agent: None for agent in self.possible_agents}
        self.current_state = {agent: None for agent in self.possible_agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, *, seed=None, options=None):
        self.agents = self.possible_agents[:]
        k_states = self.k_env.reset()

        observations = {}
        infos = {agent: {} for agent in self.agents}

        for i, agent in enumerate(self.agents):
            agent_raw_obs = k_states[i]
            
            # Generate the structured TurnBatch for this specific agent
            batch = encode_turn(agent_raw_obs.observation, self.env_cfg)
            self.current_contexts[agent] = batch.contexts
            self.current_state[agent] = batch.state

            observations[agent] = self.obs_processor.process(batch, self.env_cfg)

        return observations, infos

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # 1. Prepare actions for the Kaggle Environment (List format)
        k_actions = []
        for agent in self.possible_agents:
            if agent in actions:
                k_action = self.act_processor.process(
                    actions[agent], 
                    self.current_contexts[agent], 
                    self.current_state[agent]
                )
                k_actions.append(k_action)
            else:
                # Provide a no-op/None for dead agents
                k_actions.append(None)

        # 2. Step the underlying environment
        k_states = self.k_env.step(k_actions)

        # 3. Prepare PettingZoo return dictionaries
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for i, agent in enumerate(self.possible_agents):
            if agent not in self.agents:
                continue

            agent_state = k_states[i]
            
            # Kaggle sets statuses like "ACTIVE", "DONE", "ERROR", etc.
            status = agent_state.status
            is_done = status != "ACTIVE"
            
            # Calculate returns for this agent
            rewards[agent] = float(agent_state.reward) if agent_state.reward is not None else 0.0
            terminations[agent] = is_done
            truncations[agent] = status in ["TIMEOUT", "ERROR"] # Adjust based on how Orbit Wars defines timeouts
            infos[agent] = {"status": status}

            # Process the next observation
            batch = encode_turn(agent_state.observation, self.env_cfg)
            self.current_contexts[agent] = batch.contexts
            self.current_state[agent] = batch.state
            observations[agent] = self.obs_processor.process(batch, self.env_cfg)

        # 4. Remove terminated/truncated agents from the active list
        self.agents = [agent for agent in self.agents if not (terminations[agent] or truncations[agent])]

        return observations, rewards, terminations, truncations, infos