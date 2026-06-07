import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from collections import namedtuple
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from world_model import *

try:
    from torch_geometric.data import Data
except ImportError as exc:
    Data = None
    _PYG_IMPORT_ERROR = exc
else:
    _PYG_IMPORT_ERROR = None


IGNORE_INDEX = -100
MAX_PLANETS = 60
SAMPLE_CACHE_VERSION = 3


@dataclass(frozen=True)
class GraphFeatureConfig:
    board_size: float = 100.0
    episode_steps: int = 500
    max_planets: int = MAX_PLANETS
    max_ships: float = 500.0
    max_production: float = 5.0
    max_radius: float = 5.0
    sun_radius: float = 10.0
    rotation_radius_limit: float = 50.0
    center_x: float = 50.0
    center_y: float = 50.0
    max_fleet_speed: float = 6.0
    top_k_edges: int = 10

    # Target infer
    launch_clearance: float = 0.1
    target_inference_timestep: float = 0.25
    inference_horizon: float = 50


def _require_pyg() -> None:
    if Data is None:
        raise ImportError(
            "torch_geometric is required for graph datasets. Install torch-geometric "
            "before constructing OrbitWarsReplayDataset samples."
        ) from _PYG_IMPORT_ERROR


def replay_to_filtered_dataframe(
    json_path: str | Path, 
    target_player_name: str, 
    keep_empty_turn_ratio: float = 0.2,
    action_observation_offset: int = -1
):
    with open(json_path, "r", encoding="utf-8") as f:
        replay = json.load(f)

    # =================================================================
    # BƯỚC 1: TÌM AGENT_ID CỦA NGƯỜI CHƠI MỤC TIÊU
    # =================================================================
    team_names = replay.get("info", {}).get("TeamNames", [])
    if target_player_name not in team_names:
        return pd.DataFrame()  # Người chơi này không có trong ván đấu
    
    target_agent_id = team_names.index(target_player_name)

    steps = replay.get("steps", [])
    if not steps:
        return pd.DataFrame()

    # =================================================================
    # BƯỚC 2: KIỂM TRA ĐIỀU KIỆN CHIẾN THẮNG (CHỈ HỌC TỪ WINNER)
    # =================================================================
    rewards = replay.get("rewards", [])
    
    if not rewards or len(rewards) <= target_agent_id:
        return pd.DataFrame()  
    
    valid_rewards = [r for r in rewards if r is not None]

    if not valid_rewards or rewards[target_agent_id] is None:
        return pd.DataFrame()

    target_reward = rewards[target_agent_id]
    if target_reward < max(valid_rewards):
        return pd.DataFrame()  # Không phải người cao điểm nhất -> Bỏ qua

    # =================================================================
    # BƯỚC 3: TRÍCH XUẤT HÀNH ĐỘNG CHO MULTI-LABEL
    # =================================================================
    data_rows: list[dict[str, Any]] = []
    
    for step_idx, step in enumerate(steps):
        agent_id = target_agent_id
        if agent_id >= len(step):
            continue
            
        agent_data = step[agent_id]

        obs_step_idx = step_idx + action_observation_offset
        if obs_step_idx < 0 or obs_step_idx >= len(steps):
            continue
        if agent_id >= len(steps[obs_step_idx]):
            continue

        obs = steps[obs_step_idx][agent_id].get("observation", {})
        if not obs:
            continue

        player_id = obs.get("player", agent_id)
        action = agent_data.get("action", [])

        # =================================================================
        # BƯỚC 4: XỬ LÝ EMPTY TURNS & GỘP ACTIONS
        # =================================================================
        if not action:
            # Turn này Expert KHÔNG có hành động nào
            if random.random() <= keep_empty_turn_ratio:
                data_rows.append({
                    "step": step_idx,
                    "obs_step": obs_step_idx,
                    "player_id": player_id,
                    # Multi-label: Trả về list rỗng thay vì node -1
                    "action_sources": [], 
                    "action_angles": [],
                    "action_ships": [],
                    "raw_obs": obs,
                })
            continue # Kết thúc xử lý step này

        # Nếu có action, gộp tất cả vào các danh sách (Lists)
        sources = []
        angles = []
        ships = []

        for single_act in action:
            if len(single_act) != 3:
                continue

            from_planet_id, angle, num_ships = single_act

            sources.append(from_planet_id)
            angles.append(angle)
            ships.append(num_ships)

        # Lưu 1 turn thành ĐÚNG 1 DÒNG DUY NHẤT
        data_rows.append({
            "step": step_idx,
            "obs_step": obs_step_idx,
            "player_id": player_id,
            "action_sources": sources, # List các hành tinh xuất quân
            "action_angles": angles,   # List các góc tương ứng
            "action_ships": ships,     # List số lượng quân tương ứng
            "raw_obs": obs,
        })

    return pd.DataFrame(data_rows)


Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
Fleet = namedtuple(
    "Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
)


class OrbitWarsGraphBuilder:
    """
    Builds PyG Data graphs directly from replay observations.
    """

    node_feature_dim = 21
    edge_feature_dim = 15
    global_feature_dim = 21

    def __init__(self, config: GraphFeatureConfig | None = None) -> None:
        self.config = config or GraphFeatureConfig()
        self.world_model = None

    def obs_to_data(self, raw_obs: Any, player_id: int | None = None) -> Any:
        _require_pyg()
        if player_id is None:
            player_id = _obs_get(raw_obs, "player", 0)
        player_id = int(player_id)
        raw_planets = _obs_get(raw_obs, "planets", [])
        raw_fleets = _obs_get(raw_obs, "fleets", [])
        step = _obs_get(raw_obs, "step", 0)
        ang_vel = _obs_get(raw_obs, "angular_velocity", 0.0)
        raw_init = _obs_get(raw_obs, "initial_planets", [])

        # Ép kiểu np thành python cho WorldModel
        comets_raw = _obs_get(raw_obs, "comets", []) 
        comets = _to_py(comets_raw)
        
        comet_ids_raw = _obs_get(raw_obs, "comet_planet_ids", [])
        comet_ids = set(_to_py(comet_ids_raw))

        planets = [Planet(*planet) for planet in raw_planets]
        fleets = [Fleet(*fleet) for fleet in raw_fleets]
        initial_planets = [Planet(*planet) for planet in raw_init]
        initial_by_id = {planet.id: planet for planet in initial_planets}

        world = WorldModel(
            player=player_id,
            step=step,
            planets=planets,
            fleets=fleets,
            initial_by_id=initial_by_id,
            ang_vel=ang_vel,
            comets=comets,
            comet_ids=comet_ids,
        )
        self.world_model = world

        # Traffic Map cho heuristic features về fleets
        traffic_map = defaultdict(lambda: {
            "ally_ships": 0.0, 
            "enemy_ships": 0.0, 
            "min_eta": float('inf')
        })

        for fleet in world.fleets: 
            target_planet, eta = fleet_target_planet(fleet, world.planets)
            if target_planet is None:
                continue

            src_id = fleet.from_planet_id 
            tgt_id = target_planet.id

            key = (src_id, tgt_id)
            if fleet.owner == player_id:
                traffic_map[key]["ally_ships"] += fleet.ships
            else:
                traffic_map[key]["enemy_ships"] += fleet.ships

            # Cập nhật thời gian đến nơi (ETA) của hạm đội đến SỚM NHẤT
            traffic_map[key]["min_eta"] = min(traffic_map[key]["min_eta"], eta)

        x = torch.tensor(
            [self._node_features(planet, player_id, world) for planet in planets],
            dtype=torch.float32,
        )
        if not planets:
            x = torch.zeros((0, self.node_feature_dim), dtype=torch.float32)

        edge_index, edge_attr = self._edges(planets, player_id, world, traffic_map)
        global_attr = torch.tensor(
            [self._global_features(raw_obs, planets, fleets, player_id, world)],
            dtype=torch.float32,
        )
        owner_mask = torch.tensor(
            [planet.owner == player_id for planet in planets],
            dtype=torch.bool,
        )
        source_mask = torch.tensor(
            [planet.owner == player_id and planet.ships > 0 for planet in planets],
            dtype=torch.bool,
        )
        planet_ids = torch.tensor([planet.id for planet in planets], dtype=torch.long)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_attr=global_attr,
            owner_mask=owner_mask,
            source_mask=source_mask,
            planet_ids=planet_ids,
            num_nodes=len(planets),
        )

    def _node_features(self, planet: Planet, player_id: int, world: WorldModel) -> list[float]:
        cfg = self.config
        owner = planet.owner
        pid = planet.id
        x = planet.x
        y = planet.y
        dx = x - cfg.center_x
        dy = y - cfg.center_y
        orbit_radius = math.hypot(dx, dy)
        orbit_angle = math.atan2(dy, dx)    
        base = [
            1.0 if owner == player_id else 0.0,
            1.0 if owner not in {-1, player_id} else 0.0,
            1.0 if owner == -1 else 0.0,
            x / cfg.board_size,
            y / cfg.board_size,
            planet.radius / cfg.max_radius,
            min(planet.ships, cfg.max_ships) / cfg.max_ships,
            planet.production / cfg.max_production,
            1.0 if self._is_rotating(planet) else 0.0,
            orbit_radius / cfg.board_size,
            math.sin(orbit_angle),
            math.cos(orbit_angle),
        ]

        # Heuristic Features
        # 1. THÔNG TIN TIMELINE (Dự báo tương lai)
        keep_needed = world.keep_needed_map.get(pid, 0)
        min_owned = world.min_owned_map.get(pid, 0)
        holds_full = 1.0 if world.holds_full_map.get(pid, False) else 0.0

        first_enemy_turn = world.first_enemy_map.get(pid)
        first_enemy_val = 1.0 / (1.0 + first_enemy_turn) if first_enemy_turn is not None else 0.0

        fall_turn = world.fall_turn_map.get(pid)
        fall_turn_val = 1.0 / (1.0 + fall_turn) if fall_turn is not None else 0.0

        # 2. ÁP LỰC KHÔNG GIAN (Indirect Features)
        f_inf, n_inf, e_inf = world.indirect_feature_map.get(pid, (0.0, 0.0, 0.0))
        inf_denom = max(cfg.max_production, 1.0)

        # 3. ĐIỂM YẾU CỦA ĐỊCH (Kẻ địch vừa xả quân -> hở sườn)
        is_exposed = 1.0 if pid in world.exposed_planet_ids else 0.0

        return base + [
            min(keep_needed, cfg.max_ships) / cfg.max_ships,  # Số quân CẦN GIỮ để không mất nhà
            min(min_owned, cfg.max_ships) / cfg.max_ships,    # Mức độ an toàn trong tương lai
            holds_full,                                       # Hành tinh có trụ được không?
            first_enemy_val,                                  # Bao lâu nữa địch tới?
            fall_turn_val,                                    # Bao lâu nữa thì bị chiếm?
            f_inf / inf_denom,                                # Áp lực đồng minh xung quanh
            n_inf / inf_denom,                                # Áp lực trung lập xung quanh
            e_inf / inf_denom,                                # Áp lực địch xung quanh
            is_exposed,                                       # Cờ báo hiệu mục tiêu ngon (để tấn công)
        ]


    def _edges(self, planets: list[Planet], player_id: int, world: WorldModel, traffic_map: dict) -> tuple[torch.Tensor, torch.Tensor]:
        if len(planets) <= 1:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, self.edge_feature_dim), dtype=torch.float32),
            )

        indices: list[tuple[int, int]] = []
        attrs: list[list[float]] = []
        top_k = getattr(self.config, 'top_k_edges', 10)

        for src_idx, src in enumerate(planets):
            candidates = []
            for tgt_idx, tgt in enumerate(planets):
                if src_idx == tgt_idx:
                    continue

                dist = math.hypot(tgt.x - src.x, tgt.y - src.y)
                crosses_sun = self._shot_crosses_sun(src, tgt)
                has_traffic = False
                traffic = traffic_map.get((src.id, tgt.id))
                if traffic is not None and (traffic["ally_ships"] > 0 or traffic["enemy_ships"] > 0):
                    has_traffic = True
                candidates.append((tgt_idx, tgt, dist, crosses_sun, has_traffic))

            # Sắp xếp các hành tinh đích theo Khoảng cách (Gần nhất đứng trước)
            candidates.sort(key=lambda item: item[2])

            edges_added = 0
            for tgt_idx, tgt, dist, crosses_sun, has_traffic in candidates:
                # QUY TẮC 1: Luôn giữ cạnh nếu đang có giao tranh/vận chuyển
                if has_traffic:
                    indices.append((src_idx, tgt_idx))
                    attrs.append(self._edge_features(src, tgt, player_id, world, traffic_map))
                    continue # Bỏ qua biến đếm edges_added
                # QUY TẮC 2: Cắt bỏ các cạnh vô dụng đâm qua mặt trời
                if crosses_sun:
                    continue
                # QUY TẮC 3: Giữ lại Top-K hành tinh gần nhất để tạo Local Graph
                if edges_added < top_k:
                    indices.append((src_idx, tgt_idx))
                    attrs.append(self._edge_features(src, tgt, player_id, world, traffic_map))
                    edges_added += 1

        edge_index = torch.tensor(indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attrs, dtype=torch.float32)
        return edge_index, edge_attr

    def _edge_features(self, src: Planet, tgt: Planet, player_id: int, world: WorldModel, traffic_map: dict) -> list[float]:
        cfg = self.config
        dx = tgt.x - src.x
        dy = tgt.y - src.y
        dist = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        src_owner = src.owner
        tgt_owner = tgt.owner
        base = [
            dx / cfg.board_size,
            dy / cfg.board_size,
            dist / cfg.board_size,
            math.sin(angle),
            math.cos(angle),
            1.0 if src_owner == tgt_owner else 0.0,
            1.0 if src_owner == player_id else 0.0,
            1.0 if tgt_owner == player_id else 0.0,
            1.0 if tgt_owner not in {-1, player_id} else 0.0,
            1.0 if tgt_owner == -1 else 0.0,
            # 1.0 if self._shot_crosses_sun(src, tgt) else 0.0,
            min(src.ships, cfg.max_ships) / cfg.max_ships,
            min(tgt.ships, cfg.max_ships) / cfg.max_ships,
        ]

        # Heuristic Features
        src_id = src.id
        tgt_id = tgt.id
        # 1. TÍNH TOÁN THỜI GIAN ĐẾN NƠI (TTA - Time To Arrival)
        # aim = world.plan_shot(src_id, tgt_id, ships=1)
        # if aim is not None:
        #     angle, turns, dist_to_target, _ = aim
        #     turns_norm = turns / cfg.episode_steps
        #     dist_norm = dist_to_target / cfg.board_size
        # else:
        #     turns_norm = 1.0 
        #     dist_norm = 1.0
        
        # # 2. CHÊNH LỆCH TỐC ĐỘ PHẢN ỨNG (Mình bắn tới đó nhanh hơn hay địch nhanh hơn?)
        # my_t, enemy_t = world.reaction_times(tgt_id)
        # reaction_diff = (my_t - enemy_t) / cfg.episode_steps

        # 3. Traffic (fleet đang bay từ đâu đến đâu)
        traffic = traffic_map.get((src_id, tgt_id))
        if traffic is not None:
            ally_flying = min(traffic["ally_ships"], cfg.max_ships) / cfg.max_ships
            enemy_flying = min(traffic["enemy_ships"], cfg.max_ships) / cfg.max_ships
            eta_val = 1.0 / (1.0 + traffic["min_eta"])
        
        else:
            ally_flying = 0.0
            enemy_flying = 0.0
            eta_val = 0.0

        return base + [
            # turns_norm,    # Số turn thực tế để bay tới
            # dist_norm,     # Khoảng cách va chạm thực tế (trừ đi bán kính đích)
            # reaction_diff, # Lợi thế chiến thuật về vị trí địa lý
            ally_flying,  # Quân ta đang bay từ src -> tgt
            enemy_flying, # Quân địch đang bay từ src -> tgt
            eta_val       # Mức độ khẩn cấp (Hạm đội đầu tiên bao giờ tới?)
        ]

    def _global_features(
        self,
        raw_obs: Any,
        planets: list[Planet],
        fleets: list[Planet],
        player_id: int,
        world: WorldModel
    ) -> list[float]:
        cfg = self.config
        my_planets = world.my_planets
        enemy_planets = world.enemy_planets
        neutral_planets = world.neutral_planets
        my_fleets = [fleet for fleet in fleets if fleet.owner == player_id]
        enemy_fleets = [fleet for fleet in fleets if fleet.owner != player_id]
        max_ship_cap = max(cfg.max_planets * cfg.max_ships, 1.0)
        max_prod_cap = max(cfg.max_planets * cfg.max_production, 1.0)

        base = [
            float(_obs_get(raw_obs, "step", 0)) / cfg.episode_steps,
            float(_obs_get(raw_obs, "angular_velocity", 0.0)),
            len(planets) / cfg.max_planets,
            len(my_planets) / cfg.max_planets,
            len(enemy_planets) / cfg.max_planets,
            len(neutral_planets) / cfg.max_planets,
            sum(planet.ships for planet in neutral_planets) / max_ship_cap,
            sum(fleet.ships for fleet in my_fleets) / max_ship_cap,
            sum(fleet.ships for fleet in enemy_fleets) / max_ship_cap,
            len(my_fleets) / max(cfg.max_planets, 1),
        ]

        # Heuristic Features
        return base + [
            world.my_total / max_ship_cap,          # Tổng quân của mình (trên hành tinh + đang bay)
            world.enemy_total / max_ship_cap,         # Tổng quân của địch
            world.max_enemy_strength / max_ship_cap,# Quân của thằng địch mạnh nhất
            world.my_prod / max_prod_cap,           # Tốc độ đẻ quân của mình
            world.enemy_prod / max_prod_cap,        # Tổng tốc độ đẻ quân của phe địch
            
            1.0 if world.is_opening else 0.0,       # Giai đoạn opening
            1.0 if world.is_early else 0.0,         # Giai đoạn mở rộng
            1.0 if world.is_late else 0.0,          # Giai đoạn đánh nhau to
            1.0 if world.is_total_war else 0.0,     # Giai đoạn All-in cuối game
            
            # THÔNG TIN FFA (Dành cho game 4 người chơi)
            1.0 if world.is_four_player else 0.0,
            world._weakest_enemy_strength / max_ship_cap if world._weakest_enemy is not None else 0.0
        ]

    def _is_rotating(self, planet: Planet) -> bool:
        cfg = self.config
        radius = math.hypot(planet.x - cfg.center_x, planet.y - cfg.center_y)
        return radius + planet.radius < cfg.rotation_radius_limit

    def _shot_crosses_sun(self, src: Planet, tgt: Planet) -> bool:
        cfg = self.config
        return (
            _point_to_segment_distance(
                (cfg.center_x, cfg.center_y),
                (src.x, src.y),
                (tgt.x, tgt.y),
            )
            < cfg.sun_radius
        )
    

class OrbitWarsReplayDataset(Dataset):
    def __init__(
        self,
        replay_paths: str | Path | Iterable[str | Path] | None = None,
        dataframe: pd.DataFrame | None = None,
        *,
        cache_path: str | Path | None = None,
        graph_builder: Any | None = None,
        action_observation_offset: int = -1,
        assume_post_action_obs: bool = False,
        skip_invalid: bool = True,
        max_planets: int = MAX_PLANETS, 
    ) -> None:
        if dataframe is None:
            paths = _normalise_paths(replay_paths)
            if not paths:
                raise ValueError("Provide replay_paths or dataframe.")
            dataframe = pd.concat(
                [
                    # Giả định bạn dùng hàm replay_to_dataframe cũ ở đây
                    replay_to_filtered_dataframe(path, target_player_name='flg', action_observation_offset=action_observation_offset)
                    for path in paths
                ],
                ignore_index=True,
            )
        self.df = dataframe.reset_index(drop=True)
        self.graph_builder = graph_builder or OrbitWarsGraphBuilder() # Tạo graph_builder mặc định bên ngoài nếu None
        self.assume_post_action_obs = assume_post_action_obs
        self.skip_invalid = skip_invalid
        self.max_planets = max_planets
        self.no_action_idx = max_planets

        # Optimized data access (cache)
        self.samples = None
        if cache_path is not None and Path(cache_path).exists():
            print("Loading sample cache")
            try:
                cached = torch.load(cache_path, weights_only=False)
            except TypeError:
                cached = torch.load(cache_path)

            if (
                isinstance(cached, dict)
                and cached.get("version") == SAMPLE_CACHE_VERSION
                and "samples" in cached
            ):
                self.samples = cached["samples"]
            else:
                print("Ignoring stale sample cache")

        if self.samples is None:
            print("Preparing samples")
            self.samples = self._prepare_samples(self.df)
            if cache_path is not None:
                torch.save(
                    {"version": SAMPLE_CACHE_VERSION, "samples": self.samples},
                    cache_path,
                )

        self.records = list(self.df.itertuples(index=False))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row_idx, labels_list = self.samples[idx]

        row = self.records[row_idx]
        raw_obs = row.raw_obs
        player_id = int(row.player_id)

        # Chuyển raw_obs thành PyG Data
        data = self.graph_builder.obs_to_data(raw_obs, player_id)
        num_nodes = len(data.planet_ids)

        # =================================================================
        # KHỞI TẠO TENSOR LABEL MỚI
        # y_target: Mặc định mọi node đều là NO_ACTION_IDX
        # y_ship_pct: Mặc định là 0.0
        # =================================================================
        y_target = torch.full((num_nodes,), self.no_action_idx, dtype=torch.long)
        y_ship_pct = torch.zeros(num_nodes, dtype=torch.float32)

        for act in labels_list:
            s_idx = act["source_idx"]
            if s_idx < num_nodes: # Đề phòng lỗi out-of-bound
                y_target[s_idx] = act["target_idx"]
                y_ship_pct[s_idx] = act["ship_pct"]

        data.y_target = y_target
        data.y_ship_pct = y_ship_pct

        return data

    def _prepare_samples(self, dataframe):
        samples = []
        n = len(dataframe)

        for i, row in enumerate(dataframe.itertuples(index=False)):
            if i % 10000 == 0:
                print(f"processed {i}/{n}")

            labels_list = self._labels_for_row(row)

            if labels_list is None:
                if self.skip_invalid:
                    continue
                labels_list = []

            if len(labels_list) == 0:
                if random.random() > 0.05:
                    continue

            samples.append((i, labels_list))

        return samples

    def _labels_for_row(self, row: dict[str, Any]) -> list[dict[str, Any]] | None:
        raw_obs = row.raw_obs
        raw_planets = _obs_get(raw_obs, "planets", [])
        
        planets = [Planet(*planet) for planet in raw_planets]
        id_to_idx = {planet.id: idx for idx, planet in enumerate(planets)}
        
        if not planets:
            return None
        
        sources = getattr(row, "action_sources", [])
        angles = getattr(row, "action_angles", [])
        ships = getattr(row, "action_ships", [])

        if isinstance(sources, float) and math.isnan(sources):
            sources, angles, ships = [], [], []

        valid_actions = []
        missing_action_source = False

        for src_id, action_angle, action_ships in zip(sources, angles, ships):
            src_id = int(src_id)
            if src_id not in id_to_idx:
                missing_action_source = True
                continue

            source_idx = id_to_idx[src_id]
            source_planet = planets[source_idx]
            
            # --- 1. TÍNH TỶ LỆ TÀU (SHIP PCT) ---
            observed_ships = float(source_planet.ships)
            available_ships = max(observed_ships, 1.0)
            ship_pct = float(np.clip(float(action_ships) / available_ships, 0.0, 1.0))

            # --- 2. DỊCH GÓC BAY SANG INDEX MỤC TIÊU ---
            best_target_idx = infer_target_index_from_angle(
                                    raw_obs=raw_obs,
                                    planets=planets, # Dùng luôn list planets đang duyệt
                                    source_idx=source_idx,
                                    action_angle=float(action_angle),
                                    action_ships=float(action_ships)
                                )

            target_idx = best_target_idx if best_target_idx is not None and best_target_idx >= 0 else IGNORE_INDEX
            valid_actions.append({
                "source_idx": source_idx,
                "target_idx": target_idx,
                "ship_pct": ship_pct,
            })

        if missing_action_source:
            return None

        return valid_actions


def infer_target_index_from_angle(
    raw_obs: Any,
    planets: list[Any], # Truyền thẳng list planets đã parse vào
    source_idx: int,
    action_angle: float,
    action_ships: float,
    cfg: Any = None,
) -> int:
    """
    Suy luận Node Index của mục tiêu bằng cách mô phỏng quỹ đạo bay.
    """
    if source_idx < 0 or source_idx >= len(planets) or len(planets) <= 1:
        return IGNORE_INDEX

    source = planets[source_idx]
    ships = float(action_ships)
    cfg = cfg or GraphFeatureConfig()
    
    if not math.isfinite(action_angle) or not math.isfinite(ships) or ships <= 0.0:
        return IGNORE_INDEX

    speed = get_fleet_speed(ships, cfg)
    dir_x = math.cos(action_angle)
    dir_y = math.sin(action_angle)
    
    # Tính tọa độ xuất phát (Viền hành tinh + Khoảng cách an toàn)
    start_x = source.x + dir_x * (source.radius + cfg.launch_clearance)
    start_y = source.y + dir_y * (source.radius + cfg.launch_clearance)

    # Đọc các thông số môi trường
    angular_velocity = float(_obs_get(raw_obs, "angular_velocity", 0.0))
    comet_ids = {int(p_id) for p_id in _obs_get(raw_obs, "comet_planet_ids", [])}
    
    horizon = cfg.inference_horizon
    timestep = max(0.05, min(1.0, float(cfg.target_inference_timestep)))

    # Kiểm tra xem có bay đâm thẳng vào mặt trời không
    sun_hit_dist = ray_circle_entry_distance(
        start_x, start_y, dir_x, dir_y, cfg.center_x, cfg.center_y, cfg.sun_radius
    )
    sun_hit_time = sun_hit_dist / speed if sun_hit_dist is not None else float('inf')

    elapsed = 0.0
    while elapsed < horizon:
        next_elapsed = min(horizon, elapsed + timestep)
        
        segment_start = (
            start_x + dir_x * speed * elapsed,
            start_y + dir_y * speed * elapsed,
        )
        segment_end = (
            start_x + dir_x * speed * next_elapsed,
            start_y + dir_y * speed * next_elapsed,
        )

        step_best: tuple[float, float, int] | None = None
        
        # Duyệt qua các hành tinh để xem có đâm trúng ai trong khoảng delta_t này không
        for idx, planet in enumerate(planets):
            if idx == source_idx:
                continue

            is_comet = planet.id in comet_ids
            future_position = predict_planet_position(
                planet, next_elapsed, cfg, angular_velocity, is_comet
            )

            hit_fraction = segment_circle_entry_fraction(
                segment_start, segment_end, 
                future_position[0], future_position[1], planet.radius
            )
            
            if hit_fraction is not None:
                hit_time = elapsed + (next_elapsed - elapsed) * hit_fraction
                hit_distance = speed * hit_time
                score = (hit_time, hit_distance, idx)
                
                # Cập nhật mục tiêu chạm đầu tiên
                if step_best is None or score < step_best:
                    step_best = score

        if step_best is not None:
            hit_time, _, target_idx = step_best
            
            # Nếu đâm mặt trời TRƯỚC KHI đâm hành tinh -> Hỏng, bỏ qua lệnh này
            if sun_hit_time <= hit_time + 1e-9:
                return IGNORE_INDEX
            return target_idx

        # Nếu chưa đâm hành tinh, nhưng lại đâm mặt trời trong step này -> Hỏng
        if sun_hit_time <= next_elapsed + 1e-9:
            return IGNORE_INDEX

        elapsed = next_elapsed

    # Bay hết horizon mà không trúng gì -> Bay ra ngoài vũ trụ
    return IGNORE_INDEX

def get_fleet_speed(ships: float, cfg: Any) -> float:
    return fleet_speed(ships)

def ray_circle_entry_distance(
    ox: float, oy: float, dx: float, dy: float, cx: float, cy: float, r: float
) -> float | None:
    """Tìm khoảng cách từ điểm xuất phát (ox, oy) đến khi chạm vào hình tròn (mặt trời)."""
    # Vector từ tâm mặt trời đến điểm xuất phát
    vx, vy = ox - cx, oy - cy
    
    b = 2.0 * (vx * dx + vy * dy)
    c = (vx**2 + vy**2) - r**2
    
    delta = b**2 - 4 * c
    if delta < 0:
        return None # Không chạm
        
    t1 = (-b - math.sqrt(delta)) / 2.0
    t2 = (-b + math.sqrt(delta)) / 2.0
    
    # Lấy điểm chạm đầu tiên ở phía trước (t > 0)
    if t1 > 0: return t1
    if t2 > 0: return t2
    return None

def segment_circle_entry_fraction(
    start: tuple[float, float], end: tuple[float, float], cx: float, cy: float, r: float
) -> float | None:
    """Kiểm tra đoạn thẳng (bước nhảy thời gian) có cắt hình tròn (hành tinh) không."""
    sx, sy = start
    ex, ey = end
    
    # Vector đoạn thẳng
    seg_dx, seg_dy = ex - sx, ey - sy
    length = math.hypot(seg_dx, seg_dy)
    if length == 0: return None
    
    # Chuẩn hóa vector hướng
    dx, dy = seg_dx / length, seg_dy / length
    
    dist = ray_circle_entry_distance(sx, sy, dx, dy, cx, cy, r)
    
    # Nếu chạm và điểm chạm nằm TRONG ĐOẠN THẲNG (t <= length)
    if dist is not None and dist <= length:
        return dist / length
    return None

def predict_planet_position(
    planet: Any, elapsed: float, cfg: Any, angular_velocity: float, is_comet: bool
) -> tuple[float, float]:
    """Dự đoán vị trí của hành tinh sau `elapsed` turns."""
    if is_comet:
        # Thiên thạch thường bay thẳng hoặc theo rule riêng của game (Giả định đứng im nếu không rõ logic)
        # Nếu bạn có logic của comet, hãy bổ sung vào đây. Tạm thời trả về vị trí cũ.
        return (planet.x, planet.y)
    
    # Hành tinh quay quanh tâm
    cx, cy = cfg.center_x, cfg.center_y
    dx, dy = planet.x - cx, planet.y - cy
    radius = math.hypot(dx, dy)
    if radius + planet.radius >= cfg.rotation_radius_limit:
        return (planet.x, planet.y)

    current_angle = math.atan2(dy, dx)
    
    new_angle = current_angle + angular_velocity * elapsed
    
    new_x = cx + radius * math.cos(new_angle)
    new_y = cy + radius * math.sin(new_angle)
    return (new_x, new_y)

def _normalise_paths(paths: str | Path | Iterable[str | Path] | None) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    normalised: list[Path] = []
    for path_like in paths:
        path = Path(path_like)
        if path.is_dir():
            normalised.extend(sorted(path.glob("*.json")))
        else:
            normalised.append(path)
    return normalised


def _obs_get(obs: Any, key: str, default: Any) -> Any:
    if isinstance(obs, dict):
        value = obs.get(key, default)
    else:
        value = getattr(obs, key, default)

    if value is None:
        return default

    return value

def _point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    segment_len_sq = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2
    if segment_len_sq == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        ((point[0] - start[0]) * (end[0] - start[0]) + (point[1] - start[1]) * (end[1] - start[1]))
        / segment_len_sq
    )
    projection = max(0.0, min(1.0, projection))
    closest_x = start[0] + projection * (end[0] - start[0])
    closest_y = start[1] + projection * (end[1] - start[1])
    return math.hypot(point[0] - closest_x, point[1] - closest_y)

def _to_py(obj: Any) -> Any:
    """Đệ quy chuyển đổi toàn bộ cấu trúc Numpy Array (kể cả object arrays) về Python List/Dict."""
    if isinstance(obj, np.ndarray):
        # KHÔNG dùng obj.tolist() vì nó thất bại với dtype=object
        # Duyệt qua từng phần tử và đệ quy ép kiểu
        return [_to_py(item) for item in obj]
    elif isinstance(obj, list):
        return [_to_py(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_to_py(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    # Các kiểu dữ liệu nguyên thủy (int, float, str...) sẽ được giữ nguyên
    return obj
