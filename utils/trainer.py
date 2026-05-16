from kaggle_environments import make

class RLTrainer:
    def __init__(self, cfg, agent):
        self.cfg = cfg
        self.agent = agent
        self.env = make(self.cfg.env.name, **self.cfg.env.env_args)

    def train_one_epoch(self, **kwargs):
        # 2. Compute loss
        # 3. optimize step
        print(f"Epoch complete.")

    def evaluate(self, num_episodes=10):
        rewards = []
        for _ in range(num_episodes):
            # Run a full episode against 'random' or 'self'
            steps = self.env.run([self.agent.act, "random"])
            rewards.append(steps[-1][0].reward)
        return sum(rewards) / len(rewards)