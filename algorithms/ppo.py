from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from policy import PolicyOutput


@dataclass(slots=True)
class SampledAction:
    target_index: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


@dataclass(slots=True)
class TransitionBatch:
    self_features: torch.Tensor
    candidate_features: torch.Tensor
    global_features: torch.Tensor
    candidate_mask: torch.Tensor
    target_index: torch.Tensor
    log_prob: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

def sample_actions(outputs: PolicyOutput, deterministic: bool) -> SampledAction:
    target_logits = safe_target_logits(outputs.target_logits)
    target_dist = Categorical(logits=target_logits)
    target_index = target_logits.argmax(dim=-1) if deterministic else target_dist.sample()

    log_prob, entropy = action_log_prob_and_entropy(outputs=outputs, target_index=target_index)
    return SampledAction(target_index=target_index, log_prob=log_prob, entropy=entropy)


def action_log_prob_and_entropy(
    outputs: PolicyOutput,
    target_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_logits = safe_target_logits(outputs.target_logits)
    target_dist = Categorical(logits=target_logits)
    target_log_prob = target_dist.log_prob(target_index)
    target_entropy = target_dist.entropy()
    return target_log_prob, target_entropy

def safe_target_logits(target_logits: torch.Tensor) -> torch.Tensor:
    invalid_rows = ~torch.isfinite(target_logits).any(dim=-1)
    if not invalid_rows.any():
        return target_logits
    safe_logits = target_logits.clone()
    safe_logits[invalid_rows, 0] = 0.0
    return safe_logits

def ppo_update(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TransitionBatch,
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    epochs: int,
    minibatch_size: int,
    device: torch.device,
) -> dict[str, float]:
    if batch.self_features.shape[0] == 0:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    self_features = batch.self_features.to(device)
    candidate_features = batch.candidate_features.to(device)
    global_features = batch.global_features.to(device)
    candidate_mask = batch.candidate_mask.to(device).bool()
    old_log_prob = batch.log_prob.to(device)
    target_index = batch.target_index.to(device)
    returns = batch.returns.to(device)
    advantages = batch.advantages.to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    size = self_features.shape[0]
    minibatch_size = min(size, max(1, minibatch_size))
    metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0

    for _ in range(epochs):
        order = torch.randperm(size, device=device)
        for start in range(0, size, minibatch_size):
            idx = order[start : start + minibatch_size]
            outputs = policy(
                self_features[idx],
                candidate_features[idx],
                global_features[idx],
                candidate_mask[idx],
            )
            new_log_prob, entropy = action_log_prob_and_entropy(
                outputs,
                target_index[idx],
            )
            ratio = (new_log_prob - old_log_prob[idx]).exp()
            policy_loss = torch.maximum(
                -advantages[idx] * ratio,
                -advantages[idx] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
            ).mean()
            value_loss = 0.5 * (returns[idx] - outputs.value).pow(2).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            metrics["loss"] += float(loss.detach().cpu())
            metrics["policy_loss"] += float(policy_loss.detach().cpu())
            metrics["value_loss"] += float(value_loss.detach().cpu())
            metrics["entropy"] += float(entropy_mean.detach().cpu())
            updates += 1
    return {key: value / max(updates, 1) for key, value in metrics.items()}



# ==================================================
# REFACTOR FOR COORDINATED POLICY
# ==================================================
@dataclass(slots=True)
class SampledAction:
    target_index: torch.Tensor # [batch_size, MAX_PLANETS]
    log_prob: torch.Tensor     # [batch_size, MAX_PLANETS]
    entropy: torch.Tensor      # [batch_size, MAX_PLANETS]

@dataclass(slots=True)
class TransitionBatch:
    planet_features: torch.Tensor # [batch_size, MAX_PLANETS, planet_feat_dim]
    global_features: torch.Tensor # [batch_size, global_feat_dim]
    actor_mask: torch.Tensor      # [batch_size, MAX_PLANETS]
    target_mask: torch.Tensor     # [batch_size, MAX_PLANETS]
    target_index: torch.Tensor    # [batch_size, MAX_PLANETS]
    log_prob: torch.Tensor        # [batch_size, MAX_PLANETS]
    returns: torch.Tensor         # [batch_size] -> One return per turn
    advantages: torch.Tensor      # [batch_size] -> One advantage per turn


def sample_actions(
    outputs: PolicyOutput, 
    target_mask: torch.Tensor, 
    deterministic: bool
) -> SampledAction:
    """Samples actions for the entire sequence of planets simultaneously."""
    safe_logits = get_masked_logits(outputs.target_logits, target_mask)
    target_dist = Categorical(logits=safe_logits)
    
    if deterministic:
        target_index = safe_logits.argmax(dim=-1)
    else:
        target_index = target_dist.sample()

    log_prob = target_dist.log_prob(target_index)
    entropy = target_dist.entropy()
    
    return SampledAction(target_index=target_index, log_prob=log_prob, entropy=entropy)

def action_log_prob_and_entropy(
    outputs: PolicyOutput,
    target_index: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-evaluates log probs and entropy for the PPO update loop."""
    safe_logits = get_masked_logits(outputs.target_logits, target_mask)
    target_dist = Categorical(logits=safe_logits)
    
    target_log_prob = target_dist.log_prob(target_index)
    target_entropy = target_dist.entropy()
    
    return target_log_prob, target_entropy

def get_masked_logits(logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """
    Masks out invalid targets by setting their logits to -1e9.
    logits shape: [batch_size, MAX_PLANETS, MAX_PLANETS]
    target_mask shape: [batch_size, MAX_PLANETS]
    """
    # Broadcast target_mask to [batch_size, 1, MAX_PLANETS] so it applies to every actor's target selection
    mask = target_mask.unsqueeze(1) 
    
    # Fill invalid targets with a highly negative number so their prob becomes ~0
    return logits.masked_fill(~mask, -1e9)

def ppo_update(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TransitionBatch,
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    epochs: int,
    minibatch_size: int,
    device: torch.device,
) -> dict[str, float]:
    
    batch_size = batch.planet_features.shape[0]
    if batch_size == 0:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    # Move to device
    planet_features = batch.planet_features.to(device)
    global_features = batch.global_features.to(device)
    actor_mask = batch.actor_mask.to(device).bool()
    target_mask = batch.target_mask.to(device).bool()
    old_log_prob = batch.log_prob.to(device)
    target_index = batch.target_index.to(device)
    returns = batch.returns.to(device)
    
    # Normalize advantages at the turn level
    advantages = batch.advantages.to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    
    minibatch_size = min(batch_size, max(1, minibatch_size))
    metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0

    for _ in range(epochs):
        # We shuffle turns, not individual planet decisions
        order = torch.randperm(batch_size, device=device)
        
        for start in range(0, batch_size, minibatch_size):
            idx = order[start : start + minibatch_size]
            
            # 1. Forward Pass
            outputs = policy(
                planet_features[idx],
                global_features[idx],
            )
            
            # 2. Evaluate Actions
            new_log_prob, entropy = action_log_prob_and_entropy(
                outputs,
                target_index[idx],
                target_mask[idx],
            )
            
            # 3. Policy Loss (Evaluated per valid actor token)
            ratio = (new_log_prob - old_log_prob[idx]).exp()
            
            # Expand the turn-level advantage to shape [minibatch_size, MAX_PLANETS]
            adv_expanded = advantages[idx].unsqueeze(1).expand_as(ratio)
            
            pg_loss1 = -adv_expanded * ratio
            pg_loss2 = -adv_expanded * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            pg_loss_unmasked = torch.maximum(pg_loss1, pg_loss2)
            
            # Apply actor_mask to zero out losses for planets we don't own/can't act
            valid_actors = actor_mask[idx]
            policy_loss = (pg_loss_unmasked * valid_actors).sum() / (valid_actors.sum() + 1e-8)
            
            # 4. Value Loss (Evaluated per turn)
            # outputs.value should be shape [minibatch_size]
            value_loss = 0.5 * (returns[idx] - outputs.value).pow(2).mean()
            
            # 5. Entropy (Evaluated per valid actor token)
            entropy_mean = (entropy * valid_actors).sum() / (valid_actors.sum() + 1e-8)
            
            # 6. Backprop
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            
            # Logging
            metrics["loss"] += float(loss.detach().cpu())
            metrics["policy_loss"] += float(policy_loss.detach().cpu())
            metrics["value_loss"] += float(value_loss.detach().cpu())
            metrics["entropy"] += float(entropy_mean.detach().cpu())
            updates += 1

    return {key: value / max(updates, 1) for key, value in metrics.items()}