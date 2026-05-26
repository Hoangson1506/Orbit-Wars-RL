import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from collections import namedtuple

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


@dataclass(frozen=True)
class GraphFeatureConfig:
    board_size: float = 100.0
    episode_steps: int = 500
    max_planets: int = 48
    max_ships: float = 500.0
    max_production: float = 5.0
    max_radius: float = 5.0
    sun_radius: float = 10.0
    rotation_radius_limit: float = 50.0
    center_x: float = 50.0
    center_y: float = 50.0
    max_fleet_speed: float = 6.0
    top_k_edges: int = 10


def _require_pyg() -> None:
    if Data is None:
        raise ImportError(
            "torch_geometric is required for graph datasets. Install torch-geometric "
            "before constructing OrbitWarsReplayDataset samples."
        ) from _PYG_IMPORT_ERROR


def replay_to_dataframe(json_path: str | Path, action_observation_offset: int = -1) -> pd.DataFrame:
    """
    Parses a single episode JSON and returns a Pandas DataFrame,
    preserving the raw observation dict for downstream feature preparation.

    Kaggle replay rows store the action on the row after the observation that
    produced it. The default offset therefore pairs an action with the previous
    observation for the same agent, which is the state the policy actually saw.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        replay = json.load(f)

    steps = replay.get("steps", [])
    data_rows: list[dict[str, Any]] = []
    for step_idx, step in enumerate(steps):
        for agent_id, agent_data in enumerate(step):
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

            deductions = {}
            simulated_fleets = []

            if not action:
                # Nếu turn này Expert không làm gì cả -> Thêm row "End Turn"
                data_rows.append({
                    "step": step_idx,
                    "obs_step": obs_step_idx,
                    "player_id": player_id,
                    "action_source": -1,       # -1 đại diện cho End Turn Dummy Node
                    "action_angle": 0.0,
                    "action_ships": 0.0,
                    "ship_deductions": {},     # Chưa trừ quân nào
                    "simulated_fleets": [],  # Simulate hành động bắn
                    "raw_obs": obs,
                })
                continue

            for single_act in action:
                if len(single_act) != 3:
                    continue

                from_planet_id, angle, num_ships = single_act

                data_rows.append({
                        "step": step_idx,
                        "obs_step": obs_step_idx,
                        "player_id": player_id,
                        "action_source": from_planet_id,
                        "action_angle": angle,
                        "action_ships": num_ships,
                        "ship_deductions": dict(deductions), # Copy deductions TẠI THỜI ĐIỂM NÀY
                        "simulated_fleets": list(simulated_fleets),
                        "raw_obs": obs,
                    })
                
                deductions[str(int(from_planet_id))] = deductions.get(from_planet_id, 0) + num_ships
                simulated_fleets.append({
                    "from_planet_id": from_planet_id,
                    "angle": angle,
                    "ships": num_ships,
                    "owner": player_id
                })

            if action:
                data_rows.append({
                        "step": step_idx,
                        "obs_step": obs_step_idx,
                        "player_id": player_id,
                        "action_source": -1,
                        "action_angle": 0.0,
                        "action_ships": 0.0,
                        "ship_deductions": dict(deductions), # Gửi kèm tổng deductions
                        "simulated_fleets": list(simulated_fleets),
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

    def obs_to_data(self, raw_obs: Any, player_id: int | None = None, deductions: dict = None, simulated_fleets: list[dict] = None) -> Any:
        _require_pyg()
        player_id = _obs_get(raw_obs, "player", 0)
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

        # Trừ quân trên hành tinh có hành động để giả lập thực hiện hành động (Do observation là trước khi thực hiện hành động)
        if deductions is not None and len(deductions) > 0:
            for p in planets:
                if p.id in deductions:
                    p.ships = max(0.0, float(p.ships) - deductions[p.id])

        # Giả lập hạm đội bay từ hành tinh thực hiện hành động
        if simulated_fleets is not None and len(simulated_fleets) > 0:
            planet_by_id = {p.id: p for p in planets}
            
            for sim_fl in simulated_fleets:
                src_planet = planet_by_id.get(sim_fl["from_planet_id"])
                if src_planet:
                    ang = sim_fl["angle"]
                    
                    # Epsilon trick: Dịch hạm đội ra xa tâm hành tinh một chút 
                    # để WorldModel không nghĩ là hạm đội đang đâm vào nhà chính.
                    eps = 1e-4
                    fx = src_planet.x + math.cos(ang) * eps
                    fy = src_planet.y + math.sin(ang) * eps
                    
                    mock_fleet = Fleet(
                        id=-1,
                        owner=sim_fl["owner"],
                        ships=sim_fl["ships"],
                        angle=ang,
                        x=fx,
                        y=fy,
                        from_planet_id=sim_fl["from_planet_id"]
                    )
                    fleets.append(mock_fleet)


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

        # --- BẮT ĐẦU PHẦN THÊM DUMMY NODE (END TURN) ---
        # 1. Thêm 1 node chứa toàn số 0 vào x
        dummy_x = torch.zeros((1, self.node_feature_dim), dtype=torch.float32)
        x = torch.cat([x, dummy_x], dim=0)

        # 2. Cập nhật các mask để cho phép mô hình được quyền chọn Dummy Node
        dummy_owner = torch.tensor([True], dtype=torch.bool)
        owner_mask = torch.cat([owner_mask, dummy_owner], dim=0)

        dummy_source_mask = torch.tensor([True], dtype=torch.bool) # Lúc nào cũng có quyền dừng!
        source_mask = torch.cat([source_mask, dummy_source_mask], dim=0)

        dummy_id = torch.tensor([-1], dtype=torch.long)
        planet_ids = torch.cat([planet_ids, dummy_id], dim=0)
        # --- KẾT THÚC PHẦN THÊM DUMMY NODE ---

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_attr=global_attr,
            owner_mask=owner_mask,
            source_mask=source_mask,
            planet_ids=planet_ids,
            num_nodes=len(planets) + 1,
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
    """
    Imitation-learning dataset for pointer-policy GNNs.

    Each sample is a PyG Data object with graph tensors and labels:
      y_source: local node index of the launching planet
      y_angle: angle that the data produce
      y_ship_pct: action_ships divided by source ships in the policy observation
    """

    def __init__(
        self,
        replay_paths: str | Path | Iterable[str | Path] | None = None,
        dataframe: pd.DataFrame | None = None,
        *,
        cache_path: str | Path | None = None,
        graph_builder: OrbitWarsGraphBuilder | None = None,
        action_observation_offset: int = -1,
        assume_post_action_obs: bool = False,
        skip_invalid: bool = True,
    ) -> None:
        if dataframe is None:
            paths = _normalise_paths(replay_paths)
            if not paths:
                raise ValueError("Provide replay_paths or dataframe.")
            dataframe = pd.concat(
                [
                    replay_to_dataframe(path, action_observation_offset=action_observation_offset)
                    for path in paths
                ],
                ignore_index=True,
            )
        self.df = dataframe.reset_index(drop=True)
        self.graph_builder = graph_builder or OrbitWarsGraphBuilder()
        self.assume_post_action_obs = assume_post_action_obs
        self.skip_invalid = skip_invalid

        # Optimized data access (cache)
        if cache_path is not None and Path(cache_path).exists():
            print("Loading sample cache")
            self.samples = torch.load(cache_path)

        else:
            print("Preparing samples")
            self.samples = self._prepare_samples(self.df)

            if cache_path is not None:
                torch.save(self.samples, cache_path)

        self.records = list(self.df.itertuples(index=False))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row_idx, labels = self.samples[idx]

        row = self.records[row_idx]

        raw_obs = row.raw_obs
        player_id = int(row.player_id)
        sim_fleets = row.simulated_fleets if hasattr(row, 'simulated_fleets') else []
        deductions = row.ship_deductions if hasattr(row, 'ship_deductions') else {}

        data = self.graph_builder.obs_to_data(raw_obs, player_id, deductions=deductions, simulated_fleets=sim_fleets)

        data.y_source = torch.tensor([labels["source_idx"]], dtype=torch.long)
        data.y_angle = torch.tensor([[labels["angle_sin"], labels["angle_cos"]]], dtype=torch.float32)
        data.y_ship_pct = torch.tensor([labels["ship_pct"]], dtype=torch.float32)

        data.action_source_id = torch.tensor([labels["source_id"]], dtype=torch.long)
        data.action_angle = torch.tensor([labels["raw_angle"]], dtype=torch.long)
        data.action_ships = torch.tensor([row.action_ships], dtype=torch.float32)

        return data

    def _prepare_samples(self, dataframe):
        samples = []
        n = len(dataframe)

        for i, row in enumerate(dataframe.itertuples(index=False)):
            if i % 10000 == 0:
                print(f"processed {i}/{n}")

            labels = self._labels_for_row(row)

            if labels is None:
                if self.skip_invalid:
                    continue

                labels = {
                    "source_idx": IGNORE_INDEX,
                    "source_id": -1,
                    "angle_sin": 0.0,
                    "angle_cos": 1.0,
                    "raw_angle": 0.0,
                    "ship_pct": 0.0,
                }

            samples.append((i, labels))

        return samples

    def _labels_for_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        raw_obs = row.raw_obs
        raw_planets = _obs_get(raw_obs, "planets", [])
        planets = [Planet(*planet) for planet in raw_planets]
        id_to_idx = {planet.id: idx for idx, planet in enumerate(planets)}
        if not planets:
            return None

        if pd.isna(row.action_source):
            return None
        
        source_id = int(row.action_source)
        if source_id == -1:
            return {
                "source_id": -1,
                "source_idx": len(planets), # Trỏ tới Dummy Node (nằm ở cuối list Node)
                "angle_sin": 0.0,
                "angle_cos": 1.0,
                "raw_angle": 0.0,
                "ship_pct": 0.0,
            }
        if source_id not in id_to_idx:
            return None

        source_idx = id_to_idx[source_id]
        source_planet = planets[source_idx]
        action_angle = float(row.action_angle)
        action_ships = float(row.action_ships)

        prior_deductions = row.ship_deductions.get(source_id, 0.0)
        observed_ships = float(source_planet.ships)
        actual_ships_remaining = max(0.0, observed_ships - prior_deductions)
        available_ships = actual_ships_remaining + action_ships if self.assume_post_action_obs else actual_ships_remaining
        available_ships = max(available_ships, 1.0)
        ship_pct = float(np.clip(action_ships / available_ships, 0.0, 1.0))

        angle_sin = math.sin(action_angle)
        angle_cos = math.cos(action_angle)

        return {
            "source_id": source_id,
            "source_idx": source_idx,
            "angle_sin": angle_sin,
            "angle_cos": angle_cos,
            "raw_angle": action_angle,
            "ship_pct": ship_pct,
        }


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