import torch    
import torch.nn as nn
import math

from agents.base import BaseAgent

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

        expanded_global = global_hidden.unsqueeze(2).expand(-1, -1, self.candidate_count, -1)  # (B, P, C, H)
        expanded_self = self_hidden.unsqueeze(2).expand(-1, -1, self.candidate_count, -1)  # (B, P, C, H)
        joint = torch.cat([expanded_self, expanded_global, candidate_hidden], dim=-1)  # (B, P, C, H*3)
        action_logits = self.action_head(joint).squeeze(-1)  # (B, P, C)
        action_logits = action_logits.masked_fill(~candidate_mask.bool(), -1e9)  # Mask out invalid candidates

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
        global_hidden = self.global_encoder(global_features)  # (B, P, H), have been expanded to match candidate dimension
        candidate_hidden = self.candidate_encoder(candidate_features) # (B, P, C, H)

        expanded_global = global_hidden.unsqueeze(2).expand(-1, -1, self.candidate_count, -1)  # (B, P, C, H)
        expanded_self = self_hidden.unsqueeze(2).expand(-1, -1, self.candidate_count, -1)  # (B, P, C, H)
        pooled_candidates = candidate_hidden.mean(dim=2)  # (B, P, H)
        joint = torch.cat([self_hidden, global_hidden, pooled_candidates], dim=-1)  # (B, P, H*3)

        player_state = joint.mean(dim=1) # (B, H*3)
        
        state_value = self.value_head(player_state)

        return state_value
    
    def act(self, obs):
        return super().act(obs)