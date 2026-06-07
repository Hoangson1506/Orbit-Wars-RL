from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import VonMises

MAX_PLANETS = 60

try:
    from torch_geometric.nn import GINEConv, GATv2Conv, global_mean_pool
    from torch_geometric.utils import to_dense_batch
except ImportError as exc:  # pragma: no cover - exercised only when PyG is missing.
    GINEConv = None
    global_mean_pool = None
    to_dense_batch = None
    _PYG_IMPORT_ERROR = exc
else:
    _PYG_IMPORT_ERROR = None


IGNORE_INDEX = -100
NEG_INF = -1.0e9



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

def pointer_imitation_loss(
    output: Any, # GNNAgentOutput
    data: Any,
    *,
    num_ship_buckets: int = 20,
    target_weight: float = 1.0,
    ship_weight: float = 1.0,
    no_action_weight: float = 0.01
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    max_nodes = output.target_logits.size(1)
    
    no_action_idx = output.target_logits.size(-1) - 1 

    batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

    y_target_dense, _ = to_dense_batch(data.y_target, batch, fill_value=no_action_idx, max_num_nodes=max_nodes)
    y_ship_pct_dense, _ = to_dense_batch(data.y_ship_pct, batch, fill_value=0.0, max_num_nodes=max_nodes)

    valid_mask = output.valid_source_mask
    supervised_mask = valid_mask & (y_target_dense != IGNORE_INDEX)

    if not supervised_mask.any():
        zero = output.target_logits.sum() * 0.0 + output.ship_logits.sum() * 0.0
        parts = {
            "loss": zero.detach(),
            "target_loss": zero.detach(),
            "ship_loss": zero.detach(),
            "target_acc": zero.detach(),
            "active_target_acc": zero.detach(),
            "source_acc": zero.detach(),
            "source_iou": zero.detach(),
            "ship_error": zero.detach(),
        }
        return zero, parts

    pred_target_logits = output.target_logits[supervised_mask] # [N_valid, max_planets + 1]
    pred_ship_logits = output.ship_logits[supervised_mask]     # [N_valid, num_ship_buckets]

    y_target = y_target_dense[supervised_mask].long()
    y_ship_pct = y_ship_pct_dense[supervised_mask].float()

    num_classes = pred_target_logits.size(-1)
    ce_weights = torch.ones(num_classes, device=pred_target_logits.device)
    ce_weights[no_action_idx] = no_action_weight
    target_loss = F.cross_entropy(pred_target_logits, y_target, weight=ce_weights)

    is_acting = (y_target != no_action_idx)

    if is_acting.any():
        act_ship_logits = pred_ship_logits[is_acting]
        act_y_ship_pct = y_ship_pct[is_acting]

        target_bucket = (act_y_ship_pct * (num_ship_buckets - 1)).round().long()
        
        if random.random() < 0.01: 
            print(f"\n[DEBUG] act_y_ship_pct: {act_y_ship_pct.unique()}")
            print(f"[DEBUG] target_bucket: {target_bucket.unique()}")
            
        ship_loss = F.cross_entropy(act_ship_logits, target_bucket)
    else:
        ship_loss = pred_ship_logits.sum() * 0.0

    total = target_weight * target_loss + ship_weight * ship_loss

    with torch.no_grad():
        pred_target_class = torch.argmax(pred_target_logits, dim=-1)
        
        target_acc = (pred_target_class == y_target).float().mean()
        
        pred_acting = pred_target_class != no_action_idx
        correct_sources = (pred_acting == is_acting).float()
        source_acc = correct_sources.mean()

        pred_acting_dense = (torch.argmax(output.target_logits, dim=-1) != no_action_idx) & supervised_mask
        is_acting_dense = (y_target_dense != no_action_idx) & supervised_mask

        intersection = (pred_acting_dense & is_acting_dense).sum(dim=-1).float()
        union = (pred_acting_dense | is_acting_dense).sum(dim=-1).float()
        iou = (intersection / (union + 1e-8)).mean()

        if is_acting.any():
            act_pred_target = pred_target_class[is_acting]
            act_y_target = y_target[is_acting]
            active_target_acc = (act_pred_target == act_y_target).float().mean()
            
            # Sai số trung bình của số lượng tàu xuất đi
            pred_ship_pct_val = torch.argmax(act_ship_logits, dim=-1).float() / (num_ship_buckets - 1)
            ship_error = (pred_ship_pct_val - act_y_ship_pct).abs().mean()
        else:
            active_target_acc = torch.tensor(0.0, device=pred_target_logits.device)
            ship_error = torch.tensor(0.0, device=pred_target_logits.device)

    parts = {
        "loss": total.detach(),
        "target_loss": target_loss.detach(),
        "ship_loss": ship_loss.detach(),
        "target_acc": target_acc.detach(),
        "active_target_acc": active_target_acc.detach(),
        "source_acc": source_acc.detach(),
        "source_iou": iou.detach(),
        "ship_error": ship_error.detach(),
    }
    return total, parts

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


def _policy_index(logits: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
    if deterministic:
        return logits.argmax(dim=-1)
    return torch.distributions.Categorical(logits=logits).sample()
