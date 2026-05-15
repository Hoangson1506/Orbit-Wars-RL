import os
import shutil
import time
import sys
import torch
import random
import numpy as np
from torch.cuda.amp import GradScaler
from omegaconf import OmegaConf

from utils.logger import Logger
from utils.registry import build_model, build_loss
from utils.trainer import RLTrainer

# MODELS AND LOSSES
from agents import NearestPlanetAgent

if __name__ == "__main__":
    #-----------------------------------------------------------------------------#
    # Setup                                                                       #
    #-----------------------------------------------------------------------------#
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", "base.yaml")
    config = OmegaConf.load(config_path)
    print("="*60)
    print("Experiment Config")
    print(OmegaConf.to_yaml(config))
    print("="*60)

    if os.name == "nt":
        config.training.num_workers = 0

    config.training.device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = config.training.seed
    random.seed(seed)
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = config.training.cudnn_benchmark
        torch.backends.cudnn.deterministic = config.training.cudnn_deterministic

    model_path = f"{config.model.model_path}/{config.model.model_name}/{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(model_path, exist_ok=True)
    shutil.copyfile(os.path.abspath(__file__), "{}/train.py".format(model_path))
    # shutil.copyfile(os.path.join(script_dir, "data", "transforms.py"), "{}/transforms.py".format(model_path))
    shutil.copyfile(config_path, "{}/config.yaml".format(model_path))
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))


    #-----------------------------------------------------------------------------#
    # Agent                                                                       #
    #-----------------------------------------------------------------------------#
    agent = build_model(config)

    if config.training.grad_checkpointing:
        agent.set_grad_checkpointing(True)
    
    if config.training.checkpoint_start is not None:
        print("Start from:", config.training.checkpoint_start)
        ckpt = torch.load(config.training.checkpoint_start, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        agent.load_state_dict(state_dict, strict=False)

    agent = agent.to(config.training.device)

    #-----------------------------------------------------------------------------#
    # Trainer Setup & Execution                                                   #
    #-----------------------------------------------------------------------------#
    trainer = RLTrainer(config, agent)
    scaler = GradScaler(enabled=config.training.get("use_amp", False))

    best_reward = -float("inf")
    epochs = config.training.epochs

    print("\n" + "="*60)
    print(f"Starting Training for {epochs} Epochs")
    print("="*60)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # 1. Collect Experience & Update Weights (The RL equivalent of train_one_epoch)
        train_stats = trainer.train_one_epoch(scaler=scaler)
        
        # 2. Evaluate against baselines (e.g., 'random', previous versions)
        if epoch % config.eval.eval_every_n_epoch == 0:
            eval_reward = trainer.evaluate(num_episodes=config.eval.eval_episodes)
            
            # 3. Logging & Checkpointing
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch}/{epochs} | Time: {epoch_time:.1f}s | "
                  f"Eval Reward: {eval_reward:.2f}")

            # # Save latest checkpoint
            # checkpoint_state = {
            #     "epoch": epoch,
            #     "state_dict": agent.state_dict(),
            #     "optimizer": trainer.optimizer.state_dict(),
            #     "eval_reward": eval_reward
            # }
            # torch.save(checkpoint_state, os.path.join(model_path, "latest_ckpt.pth"))

            # # Save best checkpoint
            # if eval_reward > best_reward:
            #     best_reward = eval_reward
            #     torch.save(checkpoint_state, os.path.join(model_path, "best_ckpt.pth"))
            #     print(f" -> New Best Model Saved! (Reward: {best_reward:.2f})")
