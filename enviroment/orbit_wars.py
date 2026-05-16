import gymnasium as gym
from kaggle_environments import make

from enviroment.processor import BaseActionProcessor, BaseObservationProcessor
from enviroment.utils import encode_turn

class OrbitWarsWrapper(gym.Env):
    def __init__(self, env_cfg, obs_processor: BaseObservationProcessor, act_processor: BaseActionProcessor):
        super().__init__()
        self.env_cfg = env_cfg
        self.obs_processor = obs_processor
        self.act_processor = act_processor

        # Dynamically set spaces from strategies
        self.observation_space = self.obs_processor.get_space(self.env_cfg)
        self.action_space = self.act_processor.get_space(self.env_cfg)

        self.k_env = make("orbit_wars", debug=False)
        self.trainer = None
        self.current_contexts = []
        self.current_state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.trainer = self.k_env.train([None, "random"])
        raw_obs = self.trainer.reset()

        # Generate the structured TurnBatch using turn encoder
        batch = encode_turn(raw_obs, self.env_cfg)
        self.current_contexts = batch.contexts
        self.current_state = batch.state

        return self.obs_processor.process(batch, self.env_cfg), {}

    def step(self, action):
        k_actions = self.act_processor.process(action, self.current_contexts, self.current_state)

        raw_obs, reward, done, info = self.trainer.step(k_actions)
        reward = float(reward) if reward is not None else 0.0

        batch = encode_turn(raw_obs, self.env_cfg)
        self.current_contexts = batch.contexts
        self.current_state = batch.state

        obs = self.obs_processor.process(batch, self.env_cfg)
        return obs, reward, done, False, {}