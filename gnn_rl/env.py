from __future__ import annotations

import math
from typing import Any
import random

from kaggle_environments import make
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # Keep action translation importable without Gymnasium installed.
    gym = None


MAX_PLANETS = 60 

class OrbitWarsEnv(gym.Env):
    """
    Wrapper môi trường Kaggle Orbit Wars tương thích chuẩn Gymnasium (CleanRL).
    Hỗ trợ xuất NHIỀU hành động mỗi turn (Multi-Label) nhờ kiến trúc GNN mới.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        graph_builder: Any,
        agent_id: int = 0,
        opponent_agents: list = ["random"], 
        max_steps: int = 500,
    ) -> None:
        super().__init__()
        self.agent_id = agent_id
        self.opponent_agents = opponent_agents
        self.max_steps = max_steps
        self.graph_builder = graph_builder
        self.current_raw_obs = None
        self.current_step = 0

        # Khởi tạo môi trường Kaggle
        self.kaggle_env = make("orbit_wars", configuration={"episodeSteps": max_steps}, debug=True)

        # ==========================================
        # 1. ĐỊNH NGHĨA SPACES CHO CLEANRL
        # ==========================================
        self.action_space = spaces.Dict({
            "active_sources": spaces.MultiBinary(MAX_PLANETS),
            # Target action: Mảng [MAX_PLANETS] chứa các target index. 
            # Giới hạn từ 0 đến MAX_PLANETS (với MAX_PLANETS = NO_ACTION)
            "target": spaces.MultiDiscrete([MAX_PLANETS + 1] * MAX_PLANETS),
            "ship_pct": spaces.Box(low=0.0, high=1.0, shape=(MAX_PLANETS,), dtype=np.float32)
        })

        # Hack observation space cho CleanRL (vì ta dùng PyG Data thay vì Numpy array)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Reward shaping
        self.prev_ship_adv = 0.0
        self.prev_prod_adv = 0.0 

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            pass
        
        self.current_step = 0

        self.agent_id = random.choice([0, 1])
        opponent = random.choice(self.opponent_agents)

        setup_agents = [opponent, opponent] # Tạo mảng tạm
        setup_agents[self.agent_id] = None  # Nhét Agent của CleanRL vào vị trí ngẫu nhiên
        
        self.trainer = self.kaggle_env.train(setup_agents)
        
        self.current_raw_obs = self.trainer.reset()
        restarted_state = self._get_graph()

        wm = self.graph_builder.world_model
        self.prev_ship_adv = wm.my_total - wm.enemy_total
        self.prev_prod_adv = wm.my_prod - wm.enemy_prod
        
        return restarted_state, {}

    def step(self, action: dict) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        
        # 1. Giải mã danh sách hành động từ model
        actions_list = self._decode_action_to_move(action)
        
        # 2. Truyền danh sách hành động vào Kaggle Env
        # Kaggle cho phép actions_list rỗng [] nếu agent chọn bỏ lượt (không node nào active)
        next_raw_obs, reward, done, info = self.trainer.step(actions_list)
        self.current_raw_obs = next_raw_obs
        truncated = self.current_step >= self.max_steps
        is_done = done or truncated
        next_data = self._get_graph()

        # 3. Tính toán Reward Shaping 
        shaped_reward = self._compute_reward(bool(is_done), reward)

        return next_data, shaped_reward, bool(is_done), bool(truncated), info

    def _compute_reward(self, is_done: bool, env_reward: float) -> float:
        wm = self.graph_builder.world_model

        if is_done:
            if env_reward == 1:
                return 1.0  # Thưởng lớn tuyệt đối khi win
            elif env_reward is not None and env_reward <= 0:
                return -1.0 # Phạt khi thua hoặc hòa
            else:
                return -1.0
            
        curr_ship_adv = wm.my_total - wm.enemy_total
        curr_prod_adv = wm.my_prod - wm.enemy_prod

        delta_ship_adv = curr_ship_adv - self.prev_ship_adv
        delta_prod_adv = curr_prod_adv - self.prev_prod_adv

        self.prev_ship_adv = curr_ship_adv
        self.prev_prod_adv = curr_prod_adv

        W_SHIP = 0.002  
        W_PROD = 0.02

        shaped_reward = (delta_ship_adv * W_SHIP) + (delta_prod_adv * W_PROD)

        return float(np.clip(shaped_reward, -0.25, 0.25))

    def _decode_action_to_move(self, action: dict) -> list[list]:
        """
        Duyệt qua mảng active_sources để trích xuất NHIỀU hành động.
        Trả về list các action: [[planet_id_1, angle_1, ship_1], [planet_id_2, angle_2, ship_2], ...]
        """
        if self.current_raw_obs is None:
            return []

        active_sources = action["active_sources"]
        targets = action["target"]
        ship_pcts = action["ship_pct"]

        graph_data = self._get_graph()
        
        # Hỗ trợ cả tensor và numpy array
        planet_ids = graph_data.planet_ids
        if hasattr(planet_ids, "cpu"):
            planet_ids = planet_ids.cpu().numpy()
            
        # Tạo dictionary tra cứu số quân hiện tại nhanh chóng
        planet_ships = {int(p[0]): int(p[5]) for p in self.current_raw_obs.get("planets", [])}

        actions_list = []
        
        # Duyệt qua tối đa MAX_PLANETS hoặc độ dài thực tế của graph
        max_idx = min(len(active_sources), len(planet_ids))
        
        for i in range(max_idx):
            # Nếu node này được chọn xuất quân (True hoặc 1)
            if active_sources[i]:
                source_planet_id = int(planet_ids[i])
                
                # Bỏ qua nếu là dummy node (-1)
                if source_planet_id == -1: 
                    continue

                target_idx = int(targets[i])
                if target_idx < 0 or target_idx >= len(planet_ids):
                    continue

                target_planet_id = int(planet_ids[target_idx])
                if target_planet_id == -1 or target_planet_id == source_planet_id:
                    continue
                    
                base_ships = planet_ships.get(source_planet_id, 0)
                if base_ships == 0:
                    continue

                # Tính số lượng quân
                ship_pct = float(ship_pcts[i])
                ship_count = int(base_ships * min(max(ship_pct, 0.0), 1.0))
                ship_count = max(1, ship_count) # Gửi ít nhất 1 quân
                
                if ship_count > base_ships:
                    ship_count = base_ships

                shot_result = self.graph_builder.world_model.plan_shot(source_planet_id, target_planet_id, ship_count)
                if shot_result is None:
                    continue
                angle = float(shot_result[0])

                # Thêm vào danh sách lệnh
                actions_list.append([source_planet_id, angle, ship_count])
                
        return actions_list

    def _get_graph(self) -> Any:
        data = self.graph_builder.obs_to_data(
            self.current_raw_obs,
            self.agent_id,
        )
        return data