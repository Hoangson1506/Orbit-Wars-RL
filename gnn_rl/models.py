from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import VonMises

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
    source_logits: torch.Tensor
    angle_mu: torch.Tensor
    angle_kappa: torch.Tensor
    ship_logits: torch.Tensor
    value: torch.Tensor
    selected_source: torch.Tensor
    source_mask: torch.Tensor
    hx: torch.Tensor  # Hidden state của LSTM
    cx: torch.Tensor  # Cell state của LSTM

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

        self.dummy_embedding = nn.Parameter(torch.randn(self.hidden_dim))

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

        dummy_mask = (planet_ids == -1)
        node_h[dummy_mask] = self.dummy_embedding

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
        dropout: float = 0.2,
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
            dropout=dropout,
        )
        # self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)

        self.source_head = make_mlp(hidden_dim * 2, hidden_dim, 1, num_layers=3, dropout=dropout)
        self.angle_head = make_mlp(hidden_dim * 2, hidden_dim, 2, num_layers=3, dropout=dropout)
        self.ship_head = make_mlp(hidden_dim * 2, hidden_dim, num_ship_buckets, num_layers=3, dropout=dropout)
        self.critic_head = make_mlp(hidden_dim, hidden_dim, 1, num_layers=2, dropout=dropout)

    def forward(
        self,
        data: Any,
        *,
        source_index: torch.Tensor | None = None,
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

        batch_size = graph_context.size(0)
        dense_node_h, node_mask = to_dense_batch(node_h, batch)
        graph_per_node = graph_context[batch]

        source_logits_node = self.source_head(torch.cat([node_h, graph_per_node], dim=-1)).squeeze(-1)
        source_logits, _ = to_dense_batch(source_logits_node, batch, fill_value=NEG_INF)
        source_mask = _dense_bool_attr(data, "source_mask", batch, fallback=node_mask)
        source_mask = source_mask & node_mask
        source_mask = _force_teacher_indices(source_mask, source_index)
        source_logits = source_logits.masked_fill(~source_mask, NEG_INF)

        selected_source = _teacher_or_policy_index(
            source_logits,
            source_index,
            deterministic=deterministic,
        )
        batch_idx = torch.arange(batch_size, device=x.device)
        source_h = dense_node_h[batch_idx, selected_source]

        # Angle Prediction
        angle_out = self.angle_head(torch.cat([source_h, graph_context], dim=-1))
        mu = angle_out[:, 0]  # Radian (không giới hạn, VonMises tự wrap)
        kappa = F.softplus(angle_out[:, 1]) + 1e-3  # Đảm bảo kappa luôn dương

        ship_logits = self.ship_head(torch.cat([source_h, graph_context], dim=-1))
        
        value = self.critic_head(graph_context).squeeze(-1)

        return GNNAgentOutput(
            source_logits=source_logits,
            angle_mu=mu,
            angle_kappa=kappa,
            ship_logits=ship_logits,
            value=value,
            selected_source=selected_source,
            source_mask=source_mask,
            hx=hx, cx=cx,
        )

    @torch.no_grad()
    def act(self, data: Any, *, deterministic: bool = True) -> dict[str, torch.Tensor]:
        output = self.forward(data, deterministic=deterministic)

        if deterministic:
            angle = output.angle_mu
            ship_bucket = torch.argmax(output.ship_logits, dim=-1)
        else:
            dist = VonMises(output.angle_mu, output.angle_kappa)
            angle = dist.sample()
            ship_dist = torch.distributions.Categorical(logits=output.ship_logits)
            ship_bucket = ship_dist.sample()

        ship_pct = ship_bucket.float() / (self.num_ship_buckets - 1)

        return {
            "source": output.selected_source,
            "angle": angle,
            "ship_pct": ship_pct,
            "value": output.value,
            "hx": output.hx,
            "cx": output.cx,
        }


def pointer_imitation_loss(
    output: GNNAgentOutput,
    data: Any,
    *,
    num_ship_buckets: int = 20,
    source_weight: float = 1.0,
    angle_weight: float = 1.0,
    ship_weight: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    y_source = data.y_source.view(-1).long()
    y_angle = data.y_angle.float() # [sin, cos]
    target_angle = torch.atan2(y_angle[:, 0], y_angle[:, 1])
    y_ship_pct = data.y_ship_pct.view(-1).float()

    source_loss = _cross_entropy_or_zero(output.source_logits, y_source, ignore_index)

    valid = (y_source != ignore_index)
    if valid.any():
        dist = VonMises(output.angle_mu[valid], output.angle_kappa[valid])
        angle_loss = -dist.log_prob(target_angle[valid]).mean()

        target_bucket = (y_ship_pct[valid] * (num_ship_buckets - 1)).round().long()
        ship_loss = F.cross_entropy(output.ship_logits[valid], target_bucket)
    else:
        angle_loss = output.angle_mu.sum() * 0.0 + output.angle_kappa.sum() * 0.0
        ship_loss = output.ship_logits.sum() * 0.0

    total = source_weight * source_loss + angle_weight * angle_loss + ship_weight * ship_loss

    with torch.no_grad():
        pred_angle = output.angle_mu
        angle_error = (pred_angle - target_angle + torch.pi) % (2 * torch.pi) - torch.pi
        angle_error = angle_error.abs().mean()
        
        pred_ship_pct = torch.argmax(output.ship_logits, dim=-1).float() / (num_ship_buckets - 1)
        ship_error = (pred_ship_pct[valid] - y_ship_pct[valid]).abs().mean() if valid.any() else torch.tensor(0.0)

    parts = {
        "loss": total.detach(),
        "source_loss": source_loss.detach(),
        "angle_loss": angle_loss.detach(),
        "ship_loss": ship_loss.detach(),
        "source_acc": _accuracy(output.source_logits, y_source, ignore_index).detach(),
        "angle_error": angle_error.detach(),
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
    dense_attr, _ = to_dense_batch(attr.to(torch.bool), batch, fill_value=False)
    return dense_attr


def _teacher_or_policy_index(
    logits: torch.Tensor,
    teacher_index: torch.Tensor | None,
    *,
    deterministic: bool,
) -> torch.Tensor:
    fallback = _policy_index(logits, deterministic=deterministic)
    if teacher_index is None:
        return fallback

    teacher_index = teacher_index.to(device=logits.device, dtype=torch.long).view(-1)
    valid = (teacher_index >= 0) & (teacher_index < logits.size(1))
    safe_teacher = teacher_index.clamp(min=0, max=max(logits.size(1) - 1, 0))
    return torch.where(valid, safe_teacher, fallback)


def _force_teacher_indices(mask: torch.Tensor, teacher_index: torch.Tensor | None) -> torch.Tensor:
    if teacher_index is None:
        return mask
    teacher_index = teacher_index.to(device=mask.device, dtype=torch.long).view(-1)
    valid = (teacher_index >= 0) & (teacher_index < mask.size(1))
    if not valid.any():
        return mask

    mask = mask.clone()
    batch_idx = torch.arange(mask.size(0), device=mask.device)
    mask[batch_idx[valid], teacher_index[valid]] = True
    return mask


def _policy_index(logits: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
    if deterministic:
        return logits.argmax(dim=-1)
    return torch.distributions.Categorical(logits=logits).sample()


def _cross_entropy_or_zero(logits: torch.Tensor, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    valid = target != ignore_index
    if valid.any():
        return F.cross_entropy(logits, target, ignore_index=ignore_index)
    return logits.sum() * 0.0


def _accuracy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    pred = logits.argmax(dim=-1)
    return (pred[valid] == target[valid]).float().mean()
