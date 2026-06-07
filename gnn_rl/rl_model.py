from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
from collections import namedtuple, defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import VonMises

from world_model import *
from env import MAX_PLANETS

try:
    from torch_geometric.nn import GINEConv, GATv2Conv, global_mean_pool
    from torch_geometric.utils import to_dense_batch
    from torch_geometric.data import Data
except ImportError as exc:  # pragma: no cover - exercised only when PyG is missing.
    GINEConv = None
    global_mean_pool = None
    to_dense_batch = None
    _PYG_IMPORT_ERROR = exc
else:
    _PYG_IMPORT_ERROR = None


IGNORE_INDEX = -100
NEG_INF = -1.0e9

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

    def obs_to_data(self, raw_obs: Any, player_id: int | None = None) -> Any:
        _require_pyg()
        # player_id = _obs_get(raw_obs, "player", 0)
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


@dataclass
class GNNAgentOutput:
    target_logits: torch.Tensor
    ship_logits: torch.Tensor
    value: torch.Tensor
    valid_source_mask: torch.Tensor
    hx: torch.Tensor | None = None
    cx: torch.Tensor | None = None

def _require_pyg() -> None:
    if GINEConv is None or global_mean_pool is None or to_dense_batch is None:
        raise ImportError(
            "torch_geometric is required for GNNAgent. Install torch-geometric "
            "before constructing the model."
        ) from _PYG_IMPORT_ERROR


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    num_layers: int = 2,
    dropout: float = 0.0,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(num_layers - 1):
        layers.extend(
            [
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class GNNBackbone(nn.Module):
    """Shared node/edge/global encoder used by actor and critic heads."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        *,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _require_pyg()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim

        self.node_encoder = make_mlp(node_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.edge_encoder = make_mlp(edge_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.global_encoder = make_mlp(global_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            # conv_mlp = make_mlp(hidden_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim, concat=False, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.graph_projection = make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        global_attr: torch.Tensor,
        batch: torch.Tensor,
        planet_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_h = self.node_encoder(x)
        edge_h = self.edge_encoder(edge_attr)
        global_h = self.global_encoder(global_attr)

        node_h = node_h + global_h[batch]
        for conv, norm in zip(self.convs, self.norms):
            residual = node_h
            node_h = conv(node_h, edge_index, edge_attr=edge_h)
            node_h = norm(node_h)
            node_h = F.relu(node_h)
            node_h = self.dropout(node_h)
            node_h = node_h + residual

        pooled_h = global_mean_pool(node_h, batch)
        graph_h = self.graph_projection(torch.cat([pooled_h, global_h], dim=-1))
        return node_h, graph_h


class GNNAgent(nn.Module):
    """
    GNN actor-critic with pointer heads.

    The actor first points at a source planet, then conditions the target pointer
    and ship-percentage regressor on that source. During IL training, pass
    y_source/y_target into forward for teacher forcing.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        *,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_ship_buckets: int = 20,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_ship_buckets = num_ship_buckets

        self.backbone = GNNBackbone(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.0,
        )
        # self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.no_action_idx = MAX_PLANETS
        self.max_planets = MAX_PLANETS

        self.target_head = make_mlp(hidden_dim * 2, hidden_dim, self.max_planets + 1, num_layers=2, dropout=dropout)
        self.ship_head = make_mlp(hidden_dim * 2, hidden_dim, num_ship_buckets, num_layers=2, dropout=dropout)
        self.critic_head = make_mlp(hidden_dim, hidden_dim, 1, num_layers=2, dropout=dropout)

    def forward(
        self,
        data: Any,
        *,
        deterministic: bool = False,
        hx: torch.Tensor | None = None,
        cx: torch.Tensor | None = None,
    ) -> GNNAgentOutput:
        x = data.x
        batch = _batch_vector(data, x)
        global_attr = _global_attr(data, batch, self.backbone.global_dim)

        node_h, graph_h = self.backbone(
            x=x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            global_attr=global_attr,
            batch=batch,
            planet_ids=data.planet_ids
        )

        # if hx is None or cx is None:
        #     hx = torch.zeros_like(graph_h)
        #     cx = torch.zeros_like(graph_h)

        # hx, cx = self.lstm(graph_h, (hx, cx))
        # graph_context = hx  # Dùng bộ nhớ làm bối cảnh toàn cục
        graph_context = graph_h
        graph_per_node = graph_context[batch]
        node_ctx = torch.cat([node_h, graph_per_node], dim=-1)

        target_logits_node = self.target_head(node_ctx)
        ship_logits_node = self.ship_head(node_ctx)

        target_logits, node_mask = to_dense_batch(target_logits_node, batch, fill_value=-1e9, max_num_nodes=MAX_PLANETS)
        ship_logits, _ = to_dense_batch(ship_logits_node, batch, fill_value=0.0, max_num_nodes=MAX_PLANETS)

        valid_source_mask = _dense_bool_attr(data, "source_mask", batch, fallback=node_mask)
        valid_source_mask = valid_source_mask & node_mask

        # INVALID TARGET MASK(Padding node)
        invalid_target_mask = ~node_mask.unsqueeze(1).expand(-1, self.max_planets, -1) # [B, MAX_PLANETS, MAX_PLANETS]
        target_logits[..., :self.max_planets] = target_logits[..., :self.max_planets].masked_fill(invalid_target_mask, -1e9)

        # SELF TARGET MASK
        diag_mask = torch.eye(self.max_planets, device=target_logits.device).bool().unsqueeze(0)
        target_logits[..., :self.max_planets] = target_logits[..., :self.max_planets].masked_fill(diag_mask, -1e9)

        # INVALID SOURCE MASK
        invalid_source_mask = ~valid_source_mask
        target_logits[..., :self.max_planets] = target_logits[..., :self.max_planets].masked_fill(invalid_source_mask.unsqueeze(-1), -1e9)

        value = self.critic_head(graph_context).squeeze(-1)

        return GNNAgentOutput(
            target_logits=target_logits,
            ship_logits=ship_logits,
            value=value,
            valid_source_mask=valid_source_mask,
            hx=hx, cx=cx,
        )

    @torch.no_grad()
    def act(self, data: Any, *, deterministic: bool = True) -> dict[str, torch.Tensor]:
        output = self.forward(data, deterministic=deterministic)

        if deterministic:
            target = torch.argmax(output.target_logits, dim=-1)
            ship_bucket = torch.argmax(output.ship_logits, dim=-1)
        else:
            target_dist = torch.distributions.Categorical(logits=output.target_logits)
            target = target_dist.sample()

            ship_dist = torch.distributions.Categorical(logits=output.ship_logits)
            ship_bucket = ship_dist.sample()

        ship_pct = ship_bucket.float() / (self.num_ship_buckets - 1)
        active_sources = (target != self.no_action_idx) & output.valid_source_mask

        return {
            "active_sources": active_sources, # Mask [B, MAX_PLANETS] cho biết node nào xuất quân
            "target": target,                 # Index target [B, MAX_PLANETS]
            "ship_pct": ship_pct,             # Tỷ lệ tàu [B, MAX_PLANETS]
            "value": output.value,
            "hx": output.hx,
            "cx": output.cx,
        }


def _batch_vector(data: Any, x: torch.Tensor) -> torch.Tensor:
    batch = getattr(data, "batch", None)
    if batch is None:
        return torch.zeros((x.size(0),), dtype=torch.long, device=x.device)
    return batch


def _global_attr(data: Any, batch: torch.Tensor, global_dim: int) -> torch.Tensor:
    graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
    global_attr = getattr(data, "global_attr", None)
    if global_attr is None:
        return torch.zeros((graph_count, global_dim), dtype=torch.float32, device=batch.device)
    global_attr = global_attr.to(batch.device)
    if global_attr.dim() == 1:
        global_attr = global_attr.view(graph_count, global_dim)
    return global_attr.float()


def _dense_bool_attr(data: Any, name: str, batch: torch.Tensor, *, fallback: torch.Tensor) -> torch.Tensor:
    attr = getattr(data, name, None)
    if attr is None:
        return fallback
    dense_attr, _ = to_dense_batch(attr.to(torch.bool), batch, fill_value=False, max_num_nodes=MAX_PLANETS)
    return dense_attr

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