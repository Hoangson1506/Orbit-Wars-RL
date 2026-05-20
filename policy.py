from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(slots=True)
class PolicyOutput:
    target_logits: torch.Tensor
    value: torch.Tensor


class PlanetPolicy(nn.Module):
    def __init__(
        self,
        self_dim: int,
        candidate_dim: int,
        global_dim: int,
        candidate_count: int,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.candidate_count = candidate_count
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.target_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        self_features: torch.Tensor,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> PolicyOutput:
        self_hidden = self.self_encoder(self_features)
        global_hidden = self.global_encoder(global_features)
        candidate_hidden = self.candidate_encoder(candidate_features)
        expanded_self = self_hidden.unsqueeze(1).expand(-1, self.candidate_count, -1)
        expanded_global = global_hidden.unsqueeze(1).expand(-1, self.candidate_count, -1)
        joint = torch.cat([expanded_self, expanded_global, candidate_hidden], dim=-1)
        target_logits = self.target_head(joint).squeeze(-1)
        target_logits = target_logits.masked_fill(~candidate_mask, torch.finfo(target_logits.dtype).min)
        pooled_candidates = candidate_hidden.mean(dim=1)
        value = self.value_head(torch.cat([self_hidden, global_hidden, pooled_candidates], dim=-1)).squeeze(-1)
        return PolicyOutput(target_logits=target_logits, value=value)
    

class TransformerPlanetPolicy(nn.Module):
    def __init__(
        self,
        self_dim: int,
        candidate_dim: int,
        global_dim: int,
        candidate_count: int,
        hidden_size: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.candidate_count = candidate_count
        
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        
        # 2. Query Projection: Combines Self + Global into a single Query
        self.query_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # 3. Cross-Attention Module
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 4. Heads
        self.target_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        
        # 5. Value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        self_features: torch.Tensor,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> PolicyOutput:
        # Encode features
        self_hidden = self.self_encoder(self_features)          # [B, H]
        global_hidden = self.global_encoder(global_features)    # [B, H]
        cand_hidden = self.candidate_encoder(candidate_features)# [B, N, H]

        # 1. Build the Query
        # Combine self and global, then add a sequence dimension
        query_context = torch.cat([self_hidden, global_hidden], dim=-1)
        query = self.query_proj(query_context).unsqueeze(1)     # [B, 1, H]

        # 2. Apply Cross-Attention
        # PyTorch MHA's key_padding_mask expects True for elements to IGNORE.
        # Your candidate_mask likely uses True for VALID candidates, so we invert it (~).
        padding_mask = ~candidate_mask

        # attended_context is the attention-weighted sum of candidate features
        attended_context, _ = self.mha(
            query=query,
            key=cand_hidden,
            value=cand_hidden,
            key_padding_mask=padding_mask
        ) # [B, 1, H]

        # 3. Predict State Value
        value = self.value_head(attended_context.squeeze(1)).squeeze(-1)  # [B, H] -> [B] 

        # 4. Predict Target Logits
        # Expand the query to match the number of candidates
        expanded_query = query.expand(-1, self.candidate_count, -1) # [B, N, H]
        
        # Concatenate the Query context with each Candidate's Key
        joint = torch.cat([expanded_query, cand_hidden], dim=-1)    # [B, N, 2H]
        target_logits = self.target_head(joint).squeeze(-1)         # [B, N]

        # Apply the hard mask to ensure illegal moves are never sampled
        target_logits = target_logits.masked_fill(padding_mask, torch.finfo(target_logits.dtype).min)

        return PolicyOutput(target_logits=target_logits, value=value)