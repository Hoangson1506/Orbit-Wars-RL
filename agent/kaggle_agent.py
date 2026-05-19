import sys

import torch
import numpy as np
from kaggle_environments import make

from agent.actor_critic import Actor
from enviroment.processor import FixedActionProcessor, PaddedObservationProcessor
from enviroment.utils import encode_turn 

class PPOKaggleAgent:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = torch.device(device)
        
        # 1. Config matching your training setup
        class Config:
            board_size = 100.0
            episode_steps = 500
            candidate_count = 8
            ship_bucket_count = 8
            max_planets = 48
            max_ships = 400.0
            max_production = 5.0
            self_feature_dim = 11
            candidate_feature_dim = 14
            global_feature_dim = 8
            hidden_dim = 512
        self.config = Config()
        
        # 2. Instantiate BOTH of your processors
        self.obs_processor = PaddedObservationProcessor()
        self.act_processor = FixedActionProcessor()
        
        # 3. Load the Actor model
        self.actor = Actor(
            self.config.candidate_count, 
            self.config.self_feature_dim, 
            self.config.candidate_feature_dim, 
            self.config.global_feature_dim, 
            self.config.hidden_dim
        ).to(self.device)
        
        # Load and clean weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("actor_state_dict", checkpoint)
        clean_dict = {k.replace("module.", "").replace("actor_network.", ""): v 
                      for k, v in state_dict.items()}
        
        self.actor.load_state_dict(clean_dict, strict=False)
        self.actor.eval()

    def act(self, obs, config=None):
        """
        The standard Kaggle interface. 
        """
        with torch.no_grad():
            raw_obs = obs.get("observation", obs) if isinstance(obs, dict) else obs
            
            batch = encode_turn(raw_obs, self.config)
            
            processed = self.obs_processor.process(batch, self.config)
            
            self_feat = torch.from_numpy(processed["self_features"]).float().unsqueeze(0).to(self.device)
            cand_feat = torch.from_numpy(processed["candidates_features"]).float().unsqueeze(0).to(self.device)
            glob_feat = torch.from_numpy(processed["global_features"]).float().unsqueeze(0).to(self.device)
            mask = torch.from_numpy(processed["mask"]).bool().unsqueeze(0).to(self.device)
            
            logits = self.actor(self_feat, cand_feat, glob_feat, mask)
            print(logits)
            
            action_indices = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
        
            moves = self.act_processor.process(action_indices, batch.contexts, batch.state)

            # ==========================================
            # 🚨 KAGGLE-SAFE DEBUG PRINTS 🚨
            # ==========================================
            valid_actions = mask.sum().item()
            non_zero_actions = np.count_nonzero(action_indices)
            
            # Print to standard error to bypass Kaggle's print suppression
            print(f"PPO Step | Mask Openings: {valid_actions} | Non-Zero Choices: {non_zero_actions} | Output Moves: {len(moves)}", file=sys.stderr)
            if len(moves) > 0:
                print(f"PPO Action Executing: {moves[0]}", file=sys.stderr)
            # ==========================================
        
        return moves