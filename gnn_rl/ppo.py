# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from xml.parsers.expat import model

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions import Categorical, Bernoulli
from torch.distributions.von_mises import VonMises
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch

from rl_model import *
from env import *


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "OrbitWars-v0"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(idx, capture_video, run_name, graph_builder, opponent_agents, max_steps=500):
    def thunk():
        env = OrbitWarsEnv(
            graph_builder=graph_builder,
            opponent_agents=opponent_agents,  # Đối thủ (Có thể đổi thành bot khác)
            max_steps=max_steps
        )

        # 2. Xử lý Video (Kaggle thường không hỗ trợ rgb_array chuẩn của Gym)
        if capture_video and idx == 0:
            # LƯU Ý: Đoạn này chỉ chạy được NẾU bạn đã tự viết lại hàm render() 
            # trong class OrbitWarsCleanRLEnv để trả về numpy array hình ảnh (rgb_array).
            # Nếu chưa viết, hãy False cờ capture_video khi chạy file train.
            env.metadata["render_modes"] = ["rgb_array"]
            env.render_mode = "rgb_array" 
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        # 3. Ghi log thống kê (Cực kỳ quan trọng để code PPO lấy được Episode Return & Length)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def decode_kaggle_action(action_tensor, obs, graph_data, graph_builder):
    """Hàm phụ trợ để chuyển tensor của PPO thành mảng lệnh của game cho đối thủ"""
    actions_list = []
    # Lấy action của batch 0 (vì bot chỉ chơi 1 game 1 lúc)
    action_cpu = action_tensor.squeeze(0).cpu().numpy() 
    target_actions = action_cpu[:, 0]
    ship_bucket_actions = action_cpu[:, 1]

    planet_ids = graph_data.planet_ids.cpu().numpy()
    planet_ships = {int(p[0]): int(p[5]) for p in obs.get("planets", [])}

    max_idx = min(len(target_actions), len(planet_ids))
    for i in range(max_idx):
        source_planet_id = int(planet_ids[i])
        
        # Bỏ qua dummy node
        if source_planet_id == -1: 
            continue
            
        # Kiểm tra nếu Agent chọn "NO_ACTION"
        target_idx = int(target_actions[i])
        if target_idx == MAX_PLANETS:
            continue
            
        # Kiểm tra tính hợp lệ của target_idx
        if target_idx < 0 or target_idx >= len(planet_ids):
            continue
            
        target_planet_id = int(planet_ids[target_idx])
        
        # Bỏ qua nếu target là dummy node hoặc tự bắn vào chính mình
        if target_planet_id == -1 or target_planet_id == source_planet_id:
            continue

        base_ships = planet_ships.get(source_planet_id, 0)
        if base_ships == 0: 
            continue

        # Tính ship_pct từ bucket
        ship_pct = float(ship_bucket_actions[i]) / (20 - 1)
        ship_count = int(base_ships * min(max(ship_pct, 0.0), 1.0))
        ship_count = max(1, ship_count)
        if ship_count > base_ships: 
            ship_count = base_ships

        shot_result = graph_builder.world_model.plan_shot(source_planet_id, target_planet_id, ship_count)
        if shot_result is None:
            continue
        angle = float(shot_result[0])

        # Lệnh xuất ra: [source_id, target_id, ship_count]
        actions_list.append([source_planet_id, angle, ship_count])
            
    return actions_list

# def load_historical_agent(model_path, graph_builder, agent=None, device="cpu"):
#     """Nạp file .pth và tạo ra một hàm bot chuẩn Kaggle"""
#     # Khởi tạo lại một mạng Agent độc lập
#     historical_net = agent if agent else Agent(
#         node_dim=21, edge_dim=15, global_dim=21, 
#         hidden_dim=256, num_layers=3, num_ship_buckets=20
#     ).to(device)
    
#     if model_path is not None:
#         historical_net.load_state_dict(torch.load(model_path, map_location=device)) 
#     historical_net.eval()  

#     def kaggle_bot(obs, config):
#         player_id = obs.player 
        
#         # Tiền xử lý
#         data = graph_builder.obs_to_data(obs, player_id)
#         batch = Batch.from_data_list([data]).to(device)

#         # Chạy model (Không tính gradient)
#         with torch.no_grad():
#             action_tensor, _, _, _ = historical_net.get_action_and_value(batch, deterministic=True)

#         # Giải mã và trả về
#         return decode_kaggle_action(action_tensor, obs, data, graph_builder)

#     return kaggle_bot

def update_opponent(training_agent, opponent_agent, update_type="hard", tau=0.01):
    with torch.no_grad():
        if update_type == "hard":
            opponent_agent.load_state_dict(training_agent.state_dict())
        elif update_type == "soft":
            for target_param, param in zip(opponent_agent.parameters(), training_agent.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def create_dynamic_bot(opponent_agent, graph_builder, device="cpu"):
    def kaggle_bot(obs, config):
        player_id = obs.player 
        # print(f"Bot is acting as player: {player_id}")
        
        data = graph_builder.obs_to_data(obs, player_id)
        batch = Batch.from_data_list([data]).to(device)

        # opponent_agent luôn được giữ ở eval() mode ở vòng lặp chính
        with torch.no_grad():
            action_tensor, _, _, _ = opponent_agent.get_action_and_value(batch)

        return decode_kaggle_action(action_tensor, obs, data, graph_builder)

    return kaggle_bot

class Agent(nn.Module):
    def __init__(self, node_dim=21, edge_dim=15, global_dim=21, hidden_dim=128, num_layers=3, num_ship_buckets=20):
        super().__init__()

        self.network = GNNAgent(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_ship_buckets=num_ship_buckets
        )

        self.no_action_idx = MAX_PLANETS

    def get_value(self, x):
        output = self.network(x)
        return output.value

    def get_action_and_value(self, x, action=None, deterministic=False):
        output = self.network(x, deterministic=deterministic)

        target_probs = Categorical(logits=output.target_logits)
        ship_probs = Categorical(logits=output.ship_logits)

        if action is None and deterministic==False:
            # Giai đoạn Rollout: Môi trường cần agent đưa ra hành động
            target_action = target_probs.sample()
            ship_bucket_action = ship_probs.sample()
        elif action is None and deterministic==True:
            target_action = torch.argmax(output.target_logits, dim=-1)
            ship_bucket_action = torch.argmax(output.ship_logits, dim=-1)
        else:
            # Giai đoạn Update: PPO truyền action vào. 
            # Action shape 3D: [batch, max_nodes, 2]
            target_action = action[..., 0] 
            ship_bucket_action = action[..., 1]

        # TÍNH LOGPROB
        target_logprob = target_probs.log_prob(target_action)
        ship_logprob = ship_probs.log_prob(ship_bucket_action)

        valid_mask = output.valid_source_mask

        target_logprob = target_logprob * valid_mask

        active_action_mask = valid_mask & (target_action != self.no_action_idx)
        ship_logprob = ship_logprob * active_action_mask

        logprob = (target_logprob + ship_logprob).sum(dim=1)

        # TÍNH ENTROPY
        target_ent = target_probs.entropy() * valid_mask
        ship_ent = ship_probs.entropy() * active_action_mask

        entropy = (target_ent + ship_ent).sum(dim=1)

        if action is None:
            action_tensor = torch.stack([
                target_action.float(), 
                ship_bucket_action.float()
            ], dim=-1)
        else:
            action_tensor = action

        return action_tensor, logprob, entropy, output.value

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # agent setup
    agent = Agent(
        node_dim=21, edge_dim=15, global_dim=21, 
        hidden_dim=256, num_layers=3, num_ship_buckets=20
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    checkpoint_path = "/home/tts26/sonh/Orbit-Wars-RL/gnn_rl/artifacts/gnn_il.pt" 
    print(f"Loading weights from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        agent.load_state_dict(state_dict, strict=False)

    except Exception as e:
        print(f"Warning: Không thể load weights ({e}). Đang chạy bằng mô hình random để test luồng.")

    # opponent self-play setup
    opponent_agent = Agent(
        node_dim=21, edge_dim=15, global_dim=21, 
        hidden_dim=256, num_layers=3, num_ship_buckets=20
    ).to(device)
    opponent_agent.load_state_dict(agent.state_dict()) 
    opponent_agent.eval()

    # env setup
    graph_builder = OrbitWarsGraphBuilder()
    dynamic_bot = create_dynamic_bot(opponent_agent, graph_builder, device=device)
    envs = [make_env(
        idx=i, 
        capture_video=args.capture_video, 
        run_name=run_name, 
        graph_builder=graph_builder,
        opponent_agents=[dynamic_bot]
    )() for i in range(args.num_envs)]

    # ALGO Logic: Storage setup
    # PyG Data không thể chứa trong Tensor khởi tạo trước. Dùng List 2 chiều.
    obs_buffer = [[None for _ in range(args.num_envs)] for _ in range(args.num_steps)]

    # [num_steps, num_envs, MAX_PLANETS, 2]
    actions = torch.zeros((args.num_steps, args.num_envs, MAX_PLANETS, 2)).to(device)

    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    
    next_obs = [env.reset(seed=args.seed)[0] for env in envs] # List các PyG Data
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            for env_idx in range(args.num_envs):
                obs_buffer[step][env_idx] = next_obs[env_idx]
                
            dones[step] = next_done

            batch_next_obs = Batch.from_data_list(next_obs).to(device)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action_tensor, logprob, _, value = agent.get_action_and_value(batch_next_obs)
                values[step] = value.flatten()
            actions[step] = action_tensor
            logprobs[step] = logprob

            # 3. MANUAL ENV STEP & AUTO-RESET LOGIC
            action_cpu = action_tensor.cpu().numpy()
            next_obs_new = []
            rewards_new = []
            dones_new = []

            for env_idx, env in enumerate(envs):
                # Bung tensor [MAX_PLANETS, 3] thành dictionary cho môi trường
                target_action = action_cpu[env_idx, :, 0]
                ship_bucket_action = action_cpu[env_idx, :, 1]
                active_sources = target_action != MAX_PLANETS
                ship_pct = ship_bucket_action / (20 - 1)

                env_action = {
                    "active_sources": active_sources.astype(bool),
                    "target": target_action.astype(int),
                    "ship_pct": ship_pct,
                }
                
                o, r, term, trunc, info = env.step(env_action)
                done = term or trunc
                
                # CleanRL Auto-Reset Logic: Nếu xong game, tự động reset và lấy obs mới
                if done:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    o, _ = env.reset()
                    
                next_obs_new.append(o)
                rewards_new.append(r)
                dones_new.append(done)

            # Cập nhật next_obs và tensors
            next_obs = next_obs_new
            rewards[step] = torch.tensor(rewards_new, dtype=torch.float32).to(device).view(-1)
            next_done = torch.tensor(dones_new, dtype=torch.float32).to(device)

        # bootstrap value if not done
        with torch.no_grad():       
            batch_next_obs_final = Batch.from_data_list(next_obs).to(device)
            next_value = agent.get_value(batch_next_obs_final).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs_flat = [o for step_obs in obs_buffer for o in step_obs]
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, MAX_PLANETS, 2))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Tạo minibatch từ list
                mb_obs_list = [b_obs_flat[i] for i in mb_inds]
                b_obs_batch = Batch.from_data_list(mb_obs_list).to(device)

                # Đánh giá lại action cũ
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs_batch, b_actions[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if iteration % 10 == 0:
            torch.save(agent.state_dict(), f"models/ppo_agent_{iteration}.pth")
            update_opponent(agent, opponent_agent, update_type="hard")

    for env in envs: 
        env.close()
    writer.close()