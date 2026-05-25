from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GINEConv, global_mean_pool
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
    target_logits: torch.Tensor
    ship_pct: torch.Tensor
    value: torch.Tensor
    selected_source: torch.Tensor
    selected_target: torch.Tensor
    source_mask: torch.Tensor
    target_mask: torch.Tensor


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
            conv_mlp = make_mlp(hidden_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
            self.convs.append(GINEConv(conv_mlp, edge_dim=hidden_dim))
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_h = self.node_encoder(x)
        edge_h = self.edge_encoder(edge_attr)
        global_h = self.global_encoder(global_attr)

        node_h = node_h + global_h[batch]
        for conv, norm in zip(self.convs, self.norms):
            residual = node_h
            node_h = conv(node_h, edge_index, edge_h)
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
        dropout: float = 0.1,
        mask_target_self: bool = True,
    ) -> None:
        super().__init__()
        _require_pyg()
        self.hidden_dim = hidden_dim
        self.mask_target_self = mask_target_self
        self.backbone = GNNBackbone(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.source_head = make_mlp(hidden_dim * 2, hidden_dim, 1, num_layers=2, dropout=dropout)
        self.target_head = make_mlp(hidden_dim * 3, hidden_dim, 1, num_layers=2, dropout=dropout)
        self.ship_head = make_mlp(hidden_dim * 3, hidden_dim, 1, num_layers=3, dropout=dropout)
        self.critic_head = make_mlp(hidden_dim, hidden_dim, 1, num_layers=2, dropout=dropout)

    def forward(
        self,
        data: Any,
        *,
        source_index: torch.Tensor | None = None,
        target_index: torch.Tensor | None = None,
        deterministic: bool = True,
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
        )
        batch_size = graph_h.size(0)
        dense_node_h, node_mask = to_dense_batch(node_h, batch)
        _, max_nodes, _ = dense_node_h.shape

        graph_per_node = graph_h[batch]
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

        target_mask = _dense_bool_attr(data, "target_mask", batch, fallback=node_mask)
        target_mask = target_mask & node_mask
        if self.mask_target_self and max_nodes > 0:
            target_mask = target_mask.clone()
            target_mask[batch_idx, selected_source] = False
        target_mask = _force_teacher_indices(target_mask, target_index)

        source_context = source_h.unsqueeze(1).expand(-1, max_nodes, -1)
        graph_context = graph_h.unsqueeze(1).expand(-1, max_nodes, -1)
        target_logits = self.target_head(
            torch.cat([dense_node_h, source_context, graph_context], dim=-1)
        ).squeeze(-1)
        target_logits = target_logits.masked_fill(~target_mask, NEG_INF)

        selected_target = _teacher_or_policy_index(
            target_logits,
            target_index,
            deterministic=deterministic,
        )
        target_h = dense_node_h[batch_idx, selected_target]
        ship_pct = torch.sigmoid(self.ship_head(torch.cat([source_h, target_h, graph_h], dim=-1))).squeeze(-1)
        value = self.critic_head(graph_h).squeeze(-1)

        return GNNAgentOutput(
            source_logits=source_logits,
            target_logits=target_logits,
            ship_pct=ship_pct,
            value=value,
            selected_source=selected_source,
            selected_target=selected_target,
            source_mask=source_mask,
            target_mask=target_mask,
        )

    @torch.no_grad()
    def act(self, data: Any, *, deterministic: bool = True) -> dict[str, torch.Tensor]:
        output = self.forward(data, deterministic=deterministic)
        return {
            "source": output.selected_source,
            "target": output.selected_target,
            "ship_pct": output.ship_pct,
            "value": output.value,
        }


def pointer_imitation_loss(
    output: GNNAgentOutput,
    data: Any,
    *,
    source_weight: float = 1.0,
    target_weight: float = 1.0,
    ship_weight: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    y_source = data.y_source.view(-1).long()
    y_target = data.y_target.view(-1).long()
    y_ship_pct = data.y_ship_pct.view(-1).float()

    source_loss = _cross_entropy_or_zero(output.source_logits, y_source, ignore_index)
    target_loss = _cross_entropy_or_zero(output.target_logits, y_target, ignore_index)
    valid_ship = (y_source != ignore_index) & (y_target != ignore_index)
    if valid_ship.any():
        ship_loss = F.mse_loss(output.ship_pct[valid_ship], y_ship_pct[valid_ship])
    else:
        ship_loss = output.ship_pct.sum() * 0.0

    total = source_weight * source_loss + target_weight * target_loss + ship_weight * ship_loss
    parts = {
        "loss": total.detach(),
        "source_loss": source_loss.detach(),
        "target_loss": target_loss.detach(),
        "ship_loss": ship_loss.detach(),
        "source_acc": _accuracy(output.source_logits, y_source, ignore_index).detach(),
        "target_acc": _accuracy(output.target_logits, y_target, ignore_index).detach(),
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
