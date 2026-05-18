import torch
from tensordict.nn import TensorDictModule
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs import TransformedEnv, RewardSum, StepCounter
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.distributions import Categorical

from enviroment.orbit_wars import OrbitWarsWrapper
from enviroment.processor import PaddedObservationProcessor, FixedActionProcessor
from agents.actor_critic import Actor, Critic
from utils.registry import register_algorithm

def make_env(config, obs_processor, act_processor, device="cpu"):
    """Wraps your PettingZoo env into a TorchRL TransformedEnv."""
    # 1. Instantiate your PZ wrapper
    pz_env = OrbitWarsWrapper(config, obs_processor, act_processor)
    
    # 2. Convert to TorchRL env
    # TorchRL will automatically group agents and handle TensorDict conversions
    env = PettingZooWrapper(
        env=pz_env,
        device=device
    )
    
    # 3. Add standard transforms
    env = TransformedEnv(env)
    env.append_transform(RewardSum()) # Tracks episodic return
    env.append_transform(StepCounter()) # Tracks episode length
    
    return env

def make_mappo_models(env, config, device):
    # Extract dimensions from config (or use env specs dynamically)
    group = list(env.group_map.keys())[0]

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
        in_keys=[
            (group, "observation", "self_features"), 
            (group, "observation", "candidates_features"), 
            (group, "observation", "global_features"), 
            (group, "observation", "mask")
        ],
        out_keys=[(group, "logits")]
    )
    critic_module = TensorDictModule(
        module=raw_critic,
        in_keys=[
            (group, "observation", "self_features"), 
            (group, "observation", "candidates_features"), 
            (group, "observation", "global_features")
        ],
        out_keys=[(group, "state_value")]
    )

    # 3. Create Final PPO Operators
    actor = ProbabilisticActor(
        module=actor_module,
        spec=None, 
        in_keys=[(group, "logits")],
        out_keys=[(group, "action")],
        distribution_class=Categorical, 
        return_log_prob=True
    )

    critic = ValueOperator(
        module=critic_module,
        in_keys=[
            (group, "observation", "self_features"), 
            (group, "observation", "candidates_features"), 
            (group, "observation", "global_features")
        ],
        out_keys=[(group, "state_value")]
    )
    return actor, critic, group

def make_collector(env, actor, config, device):
    frames_per_batch = config.collector.frames_per_batch
    iters = config.collector.iters
    total_frames = frames_per_batch * iters
    
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
    )
    return collector

def make_ppo_loss(actor, critic, group, config):
    gamma = config.loss.loss_args.gamma
    lmbda = config.loss.loss_args.gae_lambda
    clip_epsilon = config.loss.loss_args.clip_epsilon
    entropy_coeff = config.loss.loss_args.entropy_coeff

    adv_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=critic,
        average_gae=True
    )

    adv_module.set_keys(
        value=(group, "state_value"),
        reward=(group, "reward"),
        done=(group, "done"),
        terminated=(group, "terminated")
    )
    
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_bonus=True,
        entropy_coeff=entropy_coeff,
        loss_critic_type="smooth_l1"
    )

    loss_module.set_keys(
        reward=(group, "reward"),
        done=(group, "done"),
        terminated=(group, "terminated"),
        action=(group, "action"),
        sample_log_prob=(group, "sample_log_prob"),
        advantage=(group, "advantage"),
        value_target=(group, "value_target")
    )
    return loss_module, adv_module


# ==========================================
# 4. MAIN BUILDER FUNCTION
# ==========================================
@register_algorithm("PPO")
def build_ppo_agent(config, obs_processor=None, act_processor=None):
    device = getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")

    # 1. Build Environment
    env = make_env(config, obs_processor, act_processor, device)
    print("\n--- TENSORDICT STRUCTURE ---")
    print(env.reset())
    print("----------------------------\n")
    # import sys; sys.exit()

    # 2. Build Models
    actor, critic, group = make_mappo_models(env, config, device)

    # 3. Build Collector
    collector = make_collector(env, actor, config, device)

    # 4. Build Loss & Advantage Modules
    loss_module, adv_module = make_ppo_loss(actor, critic, group, config)

    # 5. Package and Return
    return env, actor, critic, collector, loss_module, adv_module, group