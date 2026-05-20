from train_utils import *
from agent import build_policy

def main() -> None:
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
        opponent.sync_from(policy)
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
        if isinstance(opponent, SelfPlayOpponent) and update % cfg.self_play_update_interval == 0:
            opponent.sync_from(policy)
        if update % cfg.log_every == 0:
            print(
                f"update={update} episode_reward_mean={stats['episode_reward_mean']:.4f} "
                f"episodes={int(stats['episodes_finished'])} samples={int(stats['samples'])} "
                f"loss={metrics['loss']:.4f}"
            )
        if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
            save_checkpoint(save_dir, cfg.run_name, update, policy, optimizer, cfg)


if __name__ == "__main__":
    main()