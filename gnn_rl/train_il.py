from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable
import glob
from tqdm import tqdm, trange

import pandas as pd
import numpy as np
import torch

try:
    from torch_geometric.loader import DataLoader
except ImportError as exc:  
    DataLoader = None
    _PYG_IMPORT_ERROR = exc
else:
    _PYG_IMPORT_ERROR = None

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from gnn_rl.datasets import OrbitWarsReplayDataset
    from gnn_rl.models import GNNAgent, pointer_imitation_loss
else:
    from .datasets import OrbitWarsReplayDataset
    from .models import GNNAgent, pointer_imitation_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Orbit Wars GNN pointer policy by imitation learning.")
    parser.add_argument("replays", nargs="*", help="Replay JSON files or directories. Defaults to gnn_rl/replays.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--top-k-edges", type=int, default=12)
    parser.add_argument("--num_ship_buckets", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--source-weight", type=float, default=1.0)
    parser.add_argument("--angle-weight", type=float, default=1.0)
    parser.add_argument("--ship-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-path", default="artifacts/gnn_il.pt")
    parser.add_argument(
        "--action-observation-offset",
        type=int,
        default=-1,
        help="Replay row offset used to pair actions with observations. Kaggle replays here need -1.",
    )
    parser.add_argument("--keep-invalid", action="store_true", help="Keep rows whose source/target labels cannot be built.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_pyg_loader()
    set_seed(args.seed)
    device = resolve_device(args.device)
    replay_paths = expand_replay_paths(args.replays)

    chunk_files = sorted(glob.glob("parquet_chunks_full/*.parquet"))
    if not chunk_files:
        raise ValueError("Không tìm thấy file parquet nào!")
    
    print("Reading peek samples")
    peek_df = pd.read_parquet(chunk_files[0]).head(10) # Đọc 10 dòng cho lẹ
    print("Creating peek dataset")
    peek_dataset = OrbitWarsReplayDataset(
        replay_paths=replay_paths,
        dataframe=peek_df,
        cache_path=None, # Không cache file nháp này
        action_observation_offset=args.action_observation_offset,
        skip_invalid=not args.keep_invalid,
    )
    if len(peek_dataset) == 0:
        raise RuntimeError("Chunk đầu tiên không chứa hành động hợp lệ nào để khởi tạo model.")
    sample = peek_dataset[0]

    model = GNNAgent(
        node_dim=sample.x.size(-1),
        edge_dim=sample.edge_attr.size(-1),
        global_dim=sample.global_attr.size(-1),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_ship_buckets=args.num_ship_buckets,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val_loss = float("inf")

    print("=========Start Training==========")
    for epoch in range(1, args.epochs + 1):
        print(f"\n" + "="*50)
        print(f" START EPOCH {epoch}/{args.epochs}")
        print("="*50)

        epoch_train_losses = []
        epoch_val_losses = []

        for chunk_idx, file in enumerate(chunk_files):
            print(f"\n[Epoch {epoch}] Reading chunk {chunk_idx + 1}/{len(chunk_files)}: {file}")
            df = pd.read_parquet(file)
            cache_path = Path(file).with_suffix(".cache.pt")

            dataset = OrbitWarsReplayDataset(
                replay_paths=replay_paths,
                dataframe=df,
                cache_path=cache_path,
                action_observation_offset=args.action_observation_offset,
                skip_invalid=not args.keep_invalid,
            )

            if len(dataset) == 0:
                print("No valid chunk, skip.")
                continue

            train_dataset, val_dataset = split_dataset(dataset, args.val_split, args.seed)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
            )

            val_loader = None
            if val_dataset is not None and len(val_dataset) > 0:
                val_loader = DataLoader(
                    val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
                )

            train_stats = run_epoch(
                model, train_loader, device, optimizer=optimizer,
                source_weight=args.source_weight, angle_weight=args.angle_weight,
                ship_weight=args.ship_weight, max_grad_norm=args.max_grad_norm,
                num_ship_buckets=args.num_ship_buckets
            )

            t_loss = train_stats["loss"].item() if isinstance(train_stats["loss"], torch.Tensor) else train_stats["loss"]
            epoch_train_losses.append(t_loss)

            log_parts = [f"Chunk {chunk_idx+1}", _format_stats("train", train_stats)]

            if val_loader is not None:
                val_stats = run_epoch(
                    model, val_loader, device, optimizer=None,
                    source_weight=args.source_weight, angle_weight=args.angle_weight,
                    ship_weight=args.ship_weight, max_grad_norm=args.max_grad_norm,
                    num_ship_buckets=args.num_ship_buckets
                )
                v_loss = val_stats["loss"].item() if isinstance(val_stats["loss"], torch.Tensor) else val_stats["loss"]
                epoch_val_losses.append(v_loss)
                log_parts.append(_format_stats("val", val_stats))

            print(" | ".join(log_parts))

        if epoch_val_losses:
            avg_epoch_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        else:
            # Nếu không chia tập val, dùng train loss làm thước đo
            avg_epoch_loss = sum(epoch_train_losses) / len(epoch_train_losses)

        print(f"\n--> END EPOCH {epoch} | AVERAGE EPISODE LOSS: {avg_epoch_loss:.4f}")

        # Chỉ lưu lại Model nếu Loss trên toàn bộ các chunks của Epoch này giảm
        if avg_epoch_loss < best_val_loss:
            best_val_loss = avg_epoch_loss
            print(f"New Best Val: {best_val_loss:.4f}. Saving Checkpoint...")
            save_checkpoint(args.save_path, model, optimizer, args, epoch, best_val_loss, sample)


def run_epoch(
    model: GNNAgent,
    loader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    source_weight: float,
    angle_weight: float,
    ship_weight: float,
    max_grad_norm: float,
    num_ship_buckets: int = 20,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = defaultdict(float)
    example_count = 0

    pbar = tqdm(loader, desc="train" if is_train else "val", leave=False)

    for batch in pbar:
        batch = batch.to(device)
        batch_count = int(batch.num_graphs)
        with torch.set_grad_enabled(is_train):
            output = model(
                batch,
                source_index=batch.y_source.view(-1),
                deterministic=True,
            )
            loss, parts = pointer_imitation_loss(
                output,
                batch,
                num_ship_buckets=num_ship_buckets,
                source_weight=source_weight,
                angle_weight=angle_weight,
                ship_weight=ship_weight,
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        pbar.set_postfix(loss=float(loss.item()), 
                         source_loss=float(parts["source_loss"].item()), 
                         angle_loss=float(parts["angle_loss"].item()), 
                         ship_loss=parts["ship_loss"].item(), 
                         angle_error=parts["angle_error"].item(), 
                         ship_error=parts["ship_error"].item()
                         )

        for key, value in parts.items():
            totals[key] += float(value.item()) * batch_count
        example_count += batch_count

    return {key: value / max(example_count, 1) for key, value in totals.items()}


def split_dataset(
    dataset: OrbitWarsReplayDataset,
    val_split: float,
    seed: int,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset | None]:
    if val_split <= 0.0 or len(dataset) < 2:
        return dataset, None
    val_size = int(round(len(dataset) * val_split))
    val_size = min(max(val_size, 1), len(dataset) - 1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )
    return train_dataset, val_dataset


def expand_replay_paths(inputs: Iterable[str]) -> list[Path]:
    if not inputs:
        inputs = [str(Path(__file__).resolve().parent / "replays")]
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.json")))
        else:
            paths.append(path)
    if not paths:
        raise FileNotFoundError("No replay JSON files found.")
    return paths


def save_checkpoint(
    save_path: str,
    model: GNNAgent,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    metric_loss: float,
    sample: object,
) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metric_loss": metric_loss,
            "args": vars(args),
            "dims": {
                "node_dim": sample.x.size(-1),
                "edge_dim": sample.edge_attr.size(-1),
                "global_dim": sample.global_attr.size(-1),
            },
        },
        path,
    )


def _format_stats(prefix: str, stats: dict[str, float]) -> str:
    keys = ["loss", "source_loss", "angle_loss", "ship_loss", "source_acc", "angle_error"]
    return " ".join(f"{prefix}_{key}={stats.get(key, 0.0):.4f}" for key in keys)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _require_pyg_loader() -> None:
    if DataLoader is None:
        raise ImportError(
            "torch_geometric is required for training. Install torch-geometric "
            "before running gnn_rl/train_il.py."
        ) from _PYG_IMPORT_ERROR


if __name__ == "__main__":
    main()
