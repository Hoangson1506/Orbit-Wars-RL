import torch    
import torch.nn as nn
import math

from agent.base import BaseAgent

class Actor(BaseAgent):
    def __init__(self, candidate_count, self_dim, candidate_dim, global_dim, hidden_dim):
        super().__init__()
        self.candidate_count = candidate_count

        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, self_features, candidate_features, global_features, candidate_mask):
        self_hidden = self.self_encoder(self_features)  # (B, P, H)
        global_hidden = self.global_encoder(global_features)  # (B, P, H), have been expanded to match candidate dimension
        candidate_hidden = self.candidate_encoder(candidate_features) # (B, P, C, H)

        expanded_self = self_hidden.unsqueeze(-2).expand_as(candidate_hidden) # (B, P, C, H)
        expanded_global = global_hidden.unsqueeze(-2).expand_as(candidate_hidden)  # (B, P, C, H)
        joint = torch.cat([expanded_self, expanded_global, candidate_hidden], dim=-1)  # (B, P, C, H*3)

        # 1. Get the logits for the 8 candidates
        joint = torch.cat([expanded_self, expanded_global, candidate_hidden], dim=-1)  
        candidate_logits = self.action_head(joint).squeeze(-1) # Shape: [..., 48, 8]
        
        # 2. Create a "No Action" logit (Index 0). 
        no_action_logits = torch.zeros_like(candidate_logits[..., :1])
        
        # 3. Combine them so the network officially outputs 9 choices
        action_logits = torch.cat([no_action_logits, candidate_logits], dim=-1) # Shape: [..., 48, 9]
        
        # 4. Pad the mask so Index 0 ("No Action") is ALWAYS valid (True)
        bool_mask = candidate_mask.bool()
        always_valid = torch.ones_like(bool_mask[..., :1]) 
        full_mask = torch.cat([always_valid, bool_mask], dim=-1) # Shape: [..., 48, 9]

        # 5. Apply the aligned mask safely
        action_logits = action_logits.masked_fill(~full_mask, -1e9)

        return action_logits
        

    def act(self, obs, config=None, **kwargs):
        # Implement action selection logic here
        return []
    
class Critic(BaseAgent):
    def __init__(self, candidate_count, self_dim, candidate_dim, global_dim, hidden_dim):
        super().__init__()
        self.candidate_count = candidate_count

        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, self_features, candidate_features, global_features):
        self_hidden = self.self_encoder(self_features)  # (B, P, H)
        global_hidden = self.global_encoder(global_features)  # (B, P, H)
        candidate_hidden = self.candidate_encoder(candidate_features) # (B, P, C, H)

        pooled_candidates = candidate_hidden.mean(dim=-2)  # (B, P, H)
        joint = torch.cat([self_hidden, global_hidden, pooled_candidates], dim=-1)  # (B, P, H*3)

        player_state = joint.mean(dim=-2) # (B, H*3)
        
        state_value = self.value_head(player_state)

        return state_value
    
    def act(self, obs):
        return super().act(obs)