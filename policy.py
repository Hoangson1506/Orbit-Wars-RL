from dataclasses import dataclass
import math

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
        planet_dim: int,
        global_dim: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 3
    ) -> None:
        super().__init__()
        
        # 1. Feature Encoders
        self.planet_encoder = nn.Sequential(
            nn.Linear(planet_dim, hidden_size), 
            nn.LayerNorm(hidden_size), 
            nn.GELU()
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_size), 
            nn.LayerNorm(hidden_size), 
            nn.GELU()
        )
        
        # 2. Deep Contextualization Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            norm_first=True,  
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers, 
            enable_nested_tensor=False
        )
        self.final_ln = nn.LayerNorm(hidden_size)
        
        # 3. Action Heads (Actor via Pointer Network)
        # Extracts "sender" representation for a planet
        self.actor_query = nn.Linear(hidden_size, hidden_size) 
        # Extracts "target" representation for a planet
        self.actor_key = nn.Linear(hidden_size, hidden_size)   
        
        # 4. Value Head (Critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Initialize final layers to prevent massive gradients early on
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.constant_(self.value_head[-1].bias, 0.0)
        
        # Initialize actor heads closer to zero for flat starting probabilities
        nn.init.orthogonal_(self.actor_query.weight, gain=0.01)
        nn.init.orthogonal_(self.actor_key.weight, gain=0.01)

    def forward(
        self,
        planet_features: torch.Tensor,
        global_features: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> PolicyOutput:
        
        B, N, _ = planet_features.shape
        
        # --- 1. Embedding Stage ---
        planet_emb = self.planet_encoder(planet_features) # [B, N, H]
        
        # The global features become token index 0 (The [CLS] token)
        global_emb = self.global_encoder(global_features).unsqueeze(1) # [B, 1, H]
        
        # Build the Sequence: [Global_CLS, Planet_1, Planet_2, ..., Planet_N]
        seq = torch.cat([global_emb, planet_emb], dim=1)  # [B, N+1, H]
        
        # --- 2. Masking ---
        # If a target mask is provided, tell the transformer to ignore padded zeros
        # PyTorch transformers require True for elements to IGNORE.
        if target_mask is not None:
            # The CLS token (index 0) is always valid
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=seq.device)
            # Invert target_mask so valid planets=False, padded planets=True
            padding_mask = torch.cat([cls_mask, ~target_mask], dim=1) # [B, N+1]
        else:
            padding_mask = None

        # --- 3. Attention & Coordination ---
        seq_out = self.transformer(seq, src_key_padding_mask=padding_mask)
        seq_out = self.final_ln(seq_out)
        
        # Split the sequence back apart
        cls_out = seq_out[:, 0, :]   # [B, H]
        planet_out = seq_out[:, 1:, :] # [B, N, H]
        
        # --- 4. Critic Estimation ---
        # The value is derived from the globally-aware CLS token
        value = self.value_head(cls_out).squeeze(-1) # [B]
        
        # --- 5. Actor Target Selection ---
        # Instead of fixed target logits, planets evaluate each other.
        Q = self.actor_query(planet_out) # Who wants to send ships [B, N, H]
        K = self.actor_key(planet_out)   # Who looks like a good target [B, N, H]
        
        # Calculate Dot-Product Attention: 
        # Planet i's Query dot Planet j's Key creates the logit for i -> j
        d_k = Q.size(-1)
        target_logits = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k) # [B, N, N]
        
        return PolicyOutput(target_logits=target_logits, value=value)