from train_utils import *
from agent import build_policy
from functools import partial
import multiprocessing as mp

def train() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    opponent = build_opponent(cfg.opponent, cfg=cfg, device=device)
    envs = [OrbitWarsEnv(cfg, opponent, env_index=idx) for idx in range(cfg.ppo.num_envs)]
    next_seed = cfg.seed
    batches = []
    for env in envs:
        batches.append(env.reset(seed=next_seed))
        next_seed += 1
    policy = build_policy(cfg=cfg, device=device)
    
    if isinstance(opponent, SelfPlayOpponent):
        original_sync_mode = cfg.sync_mode
        cfg.sync_mode = "checkpoint"
        opponent.sync_from(policy)
        cfg.sync_mode = original_sync_mode

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr)
    save_dir = Path(cfg.save_dir)
    for update in range(1, cfg.ppo.total_updates + 1):
        batch, batches, next_seed, stats = collect_rollout(envs, batches, policy, cfg, device, next_seed)

        metrics = ppo_update(
            policy,
            optimizer,
            batch,
            clip_coef=cfg.ppo.clip_coef,
            ent_coef=cfg.ppo.ent_coef,
            vf_coef=cfg.ppo.vf_coef,
            max_grad_norm=cfg.ppo.max_grad_norm,
            epochs=cfg.ppo.epochs,
            minibatch_size=cfg.ppo.minibatch_size,
            device=device,
        )
        if isinstance(opponent, SelfPlayOpponent):
            opponent.sync_from(policy, update)
        if update % cfg.log_every == 0:
            print(
                f"update={update} episode_reward_mean={stats['episode_reward_mean']:.4f} "
                f"episodes={int(stats['episodes_finished'])} samples={int(stats['samples'])} "
                f"loss={metrics['loss']:.4f}"
            )
        if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
            save_checkpoint(save_dir, cfg.run_name, update, policy, optimizer, cfg)

def multiprocessing_train() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    env_fns = [partial(build_single_env, idx, cfg, "cpu") for idx in range(cfg.ppo.num_envs)]
    envs = SubprocVectorEnv(env_fns)

    next_seed = cfg.seed
    seeds = [next_seed + i for i in range(cfg.ppo.num_envs)]
    batches = envs.reset(seeds)
    next_seed += cfg.ppo.num_envs

    policy = build_policy(cfg=cfg, device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr)
    save_dir = Path(cfg.save_dir)

    if cfg.opponent == "self":
        envs.sync_opponents(policy.state_dict())
    
    for update in range(1, cfg.ppo.total_updates + 1):
        batch, batches, next_seed, stats = multiprocessing_collect_rollout(envs, batches, policy, cfg, device, next_seed)

        metrics = ppo_update(
            policy,
            optimizer,
            batch,
            clip_coef=cfg.ppo.clip_coef,
            ent_coef=cfg.ppo.ent_coef,
            vf_coef=cfg.ppo.vf_coef,
            max_grad_norm=cfg.ppo.max_grad_norm,
            epochs=cfg.ppo.epochs,
            minibatch_size=cfg.ppo.minibatch_size,
            device=device,
        )
        
        if cfg.opponent == "self" and update % cfg.self_play_update_interval == 0:
            envs.sync_opponents(policy.state_dict())
            
        if update % cfg.log_every == 0:
            print(
                f"update={update} episode_reward_mean={stats['episode_reward_mean']:.4f} "
                f"episodes={int(stats['episodes_finished'])} samples={int(stats['samples'])} "
                f"loss={metrics['loss']:.4f}"
            )
            
        if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
            save_checkpoint(save_dir, cfg.run_name, update, policy, optimizer, cfg)
            
    envs.close()
    


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    multiprocessing_train()