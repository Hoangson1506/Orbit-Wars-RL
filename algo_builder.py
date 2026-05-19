import torch
from tensordict.nn import TensorDictModule
from torchrl.envs.libs import GymWrapper
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs import TransformedEnv, RewardSum, StepCounter
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.distributions import Categorical, Independent

from enviroment.orbit_wars import OrbitWarsWrapper, OrbitWarsSingleWrapper
from enviroment.processor import PaddedObservationProcessor, FixedActionProcessor
from agent.actor_critic import Actor, Critic
from utils.registry import register_algorithm

class IndependentCategorical(Independent):
    """
    Wraps Categorical to tell TorchRL that the last batch dimension (48 planets)
    is actually part of a single joint event. This automatically sums the log_probs!
    """
    def __init__(self, logits, **kwargs):
        super().__init__(Categorical(logits=logits, **kwargs), 1)


def make_env(config, opponent_actor, obs_processor, act_processor, device="cpu"):
    """Wraps your PettingZoo env into a TorchRL TransformedEnv."""
    # 1. Instantiate your PZ wrapper
    gym_env = OrbitWarsSingleWrapper(config, opponent_actor, obs_processor, act_processor)
    
    # 2. Convert to TorchRL env
    # TorchRL will automatically group agents and handle TensorDict conversions
    env = GymWrapper(
        env=gym_env,
        device=device
    )
    
    # 3. Add standard transforms
    env = TransformedEnv(env)
    env.append_transform(RewardSum()) # Tracks episodic return
    env.append_transform(StepCounter()) # Tracks episode length
    
    return env

def make_mappo_models(env, config, device):
    cand_count = config.model.model_args.candidate_count
    self_dim = config.model.model_args.self_feature_dim
    cand_dim = config.model.model_args.candidate_feature_dim
    global_dim = config.model.model_args.global_feature_dim
    hidden_dim = config.model.model_args.hidden_dim

    # 1. Instantiate Raw Modules
    raw_actor = Actor(cand_count, self_dim, cand_dim, global_dim, hidden_dim).to(device)
    raw_critic = Critic(cand_count, self_dim, cand_dim, global_dim, hidden_dim).to(device)

    # 2. Wrap in TensorDict Modules
    actor_module = TensorDictModule(
        module=raw_actor,
        in_keys=["self_features", "candidates_features", "global_features", "mask"],
        out_keys=["logits"]
    )
    critic_module = TensorDictModule(
        module=raw_critic,
        in_keys=["self_features", "candidates_features", "global_features"],
        out_keys=["state_value"]
    )

    # 3. Create Final PPO Operators
    actor = ProbabilisticActor(
        module=actor_module,
        spec=env.action_spec, 
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=IndependentCategorical, 
        return_log_prob=True,
        log_prob_key="sample_log_prob"
    )

    critic = ValueOperator(
        module=critic_module,
        in_keys=["self_features", "candidates_features", "global_features"],
        out_keys=["state_value"]
    )
    return actor, critic

def make_collector(env, actor, config, device):
    return SyncDataCollector(
        env,
        actor,
        frames_per_batch=config.collector.frames_per_batch,
        total_frames=config.collector.frames_per_batch * config.collector.iters,
        device=device,
        storing_device=device,
    )

def make_ppo_loss(actor, critic, config):
    adv_module = GAE(
        gamma=config.loss.loss_args.gamma,
        lmbda=config.loss.loss_args.gae_lambda,
        value_network=critic,
        average_gae=True
    )

    # Flat keys
    adv_module.set_keys(
        value="state_value",
        reward="reward",
        done="done",
        terminated="terminated",
        advantage="advantage",        
        value_target="value_target"
    )
    
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=config.loss.loss_args.clip_epsilon,
        entropy_bonus=True,
        entropy_coeff=config.loss.loss_args.entropy_coeff,
        loss_critic_type="smooth_l1"
    )

    loss_module.set_keys(
        value="state_value",
        reward="reward",
        done="done",
        terminated="terminated",
        action="action",
        sample_log_prob="sample_log_prob",
        advantage="advantage",
        value_target="value_target"
    )
    return loss_module, adv_module


# ==========================================
# 4. MAIN BUILDER FUNCTION
# ==========================================
@register_algorithm("PPO")
def build_ppo_agent(config, opponent_actor, obs_processor=None, act_processor=None):
    device = getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")

    # Pass the Frozen Opponent explicitly into the Environment Maker
    env = make_env(config, opponent_actor, obs_processor, act_processor, device)

    # Build Models (No more group parsing)
    actor, critic = make_mappo_models(env, config, device)

    collector = make_collector(env, actor, config, device)
    loss_module, adv_module = make_ppo_loss(actor, critic, config)

    return env, actor, critic, collector, loss_module, adv_module