import os
import shutil
import time
import sys
import torch
import random
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from utils.logger import Logger
from utils.registry import build_model, build_loss

# MODELS AND LOSSES
from agents.base import NearestPlanetAgent
from algo_builder import build_ppo_agent
from enviroment.processor import PaddedObservationProcessor, FixedActionProcessor

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
    # Agent and Environment Setup                                                 #
    #-----------------------------------------------------------------------------#
    env, actor, critic, collector, loss_module, adv_module, group = build_ppo_agent(config, PaddedObservationProcessor(), FixedActionProcessor())
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config.training.lr)

    # if config.training.grad_checkpointing:
    #     agent.set_grad_checkpointing(True)
    
    # if config.training.checkpoint_start is not None:
    #     print("Start from:", config.training.checkpoint_start)
    #     ckpt = torch.load(config.training.checkpoint_start, map_location="cpu")
    #     state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    #     agent.load_state_dict(state_dict, strict=False)

    best_reward = -float("inf")
    epochs = config.training.epochs
    iters = config.collector.iters
    frames_per_batch = config.collector.frames_per_batch
    replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(
            frames_per_batch, device=config.training.device
        ),  # We store the frames_per_batch collected at each iteration
        sampler=SamplerWithoutReplacement(),
        batch_size=config.training.batch_size,  # We will sample minibatches of this size
    )
    pbar = tqdm(total=iters*frames_per_batch, desc="Total Frames")

    print("\n" + "="*60)
    print(f"Starting Training for {epochs} Epochs")
    print("="*60)

    for i, tensordict_data in enumerate(collector):
        tensordict_data = tensordict_data.to(config.training.device)

        with torch.no_grad():
            adv_module(tensordict_data)

        data_view = tensordict_data.reshape(-1)  
        replay_buffer.extend(data_view)

        total_loss_this_batch = 0.0
        
        num_minibatches = max(1, len(data_view) // config.training.batch_size)

        # 4. Loop over PPO Epochs
        for epoch in range(config.training.epochs):
            
            # 5. Loop over minibatches using the ReplayBuffer
            for _ in range(num_minibatches):
                minibatch = replay_buffer.sample()
                
                # Compute loss values
                loss_vals = loss_module(minibatch)
                
                loss_value = (
                    loss_vals["loss_objective"] + 
                    loss_vals["loss_critic"] + 
                    loss_vals["loss_entropy"]
                )
                
                # 6. Back propagate
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), config.training.max_grad_norm)
                
                # 7. Optimise
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss_this_batch += loss_value.item()

        # ==========================================
        # Logging & Metrics
        # ==========================================
        avg_reward = 0.0
        next_td = tensordict_data.get("next")
        if next_td is not None and (group, "episode_reward") in next_td.keys():
            dones = next_td.get((group, "done"))
            
            if dones.any():
                ep_rewards = next_td.get((group, "episode_reward"))[dones]
                avg_reward = ep_rewards.mean().item()
                pbar.set_postfix({"Avg Return": f"{avg_reward:.2f}", "Loss": f"{total_loss_this_batch:.2f}"})
        
        pbar.update(tensordict_data.numel())
        
        # ==========================================
        # Checkpointing Logic
        # ==========================================
        # Save best checkpoint (based on the highest training batch reward)
        if avg_reward > best_reward and dones.any():
            best_reward = avg_reward
            torch.save(checkpoint_state, os.path.join(model_path, f"iter_{i}_r_{avg_reward:.3f}.pth"))
            print("\n -> New Best Model Saved! (Reward: {:.2f})".format(best_reward))
            
        # Save latest checkpoint
        checkpoint_state = {
            "iteration": i,
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "eval_reward": avg_reward
        }
        torch.save(checkpoint_state, os.path.join(model_path, "latest_ckpt.pth"))

    pbar.close()
    
    print("\nTraining Complete! Saving final models...")
    torch.save(actor.state_dict(), os.path.join(model_path, "orbit_wars_actor_final.pt"))
    torch.save(critic.state_dict(), os.path.join(model_path, "orbit_wars_critic_final.pt"))
    print("Models saved successfully.")