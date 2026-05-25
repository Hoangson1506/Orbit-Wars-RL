import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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
    max_planets: int = 40
    max_ships: float = 400.0
    max_production: float = 5.0
    max_radius: float = 5.0
    sun_radius: float = 10.0
    rotation_radius_limit: float = 50.0
    center_x: float = 50.0
    center_y: float = 50.0
    max_fleet_speed: float = 6.0
    launch_clearance: float = 0.1
    target_inference_horizon: int = 130
    target_inference_timestep: float = 0.25


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
            action = agent_data.get("action")
            if not action:
                continue

            obs_step_idx = step_idx + action_observation_offset
            if obs_step_idx < 0 or obs_step_idx >= len(steps):
                continue
            if agent_id >= len(steps[obs_step_idx]):
                continue

            obs = steps[obs_step_idx][agent_id].get("observation", {})
            if not obs:
                continue

            player_id = obs.get("player", agent_id)
            for single_act in action:
                if len(single_act) != 3:
                    continue

                from_planet_id, angle, num_ships = single_act

                data_rows.append(
                    {
                        "step": step_idx,
                        "obs_step": obs_step_idx,
                        "player_id": player_id,
                        "action_source": from_planet_id,
                        "action_angle": angle,
                        "action_ships": num_ships,
                        "raw_obs": obs,
                    }
                )

    return pd.DataFrame(data_rows)


class OrbitWarsGraphBuilder:
    """
    Builds PyG Data graphs directly from replay observations.
    """

    node_feature_dim = 12
    edge_feature_dim = 13
    global_feature_dim = 12

    def __init__(self, config: GraphFeatureConfig | None = None) -> None:
        self.config = config or GraphFeatureConfig()

    def obs_to_data(self, raw_obs: Any, player_id: int | None = None) -> Any:
        _require_pyg()
        player_id = _player_id(raw_obs, player_id)
        planets = _sorted_planets(raw_obs)
        fleets = _fleets(raw_obs)

        x = torch.tensor(
            [self._node_features(planet, player_id) for planet in planets],
            dtype=torch.float32,
        )
        if not planets:
            x = torch.zeros((0, self.node_feature_dim), dtype=torch.float32)

        edge_index, edge_attr = self._edges(planets, player_id)
        global_attr = torch.tensor(
            [self._global_features(raw_obs, planets, fleets, player_id)],
            dtype=torch.float32,
        )
        owner_mask = torch.tensor(
            [_planet_owner(planet) == player_id for planet in planets],
            dtype=torch.bool,
        )
        source_mask = torch.tensor(
            [_planet_owner(planet) == player_id and _planet_ships(planet) > 0 for planet in planets],
            dtype=torch.bool,
        )
        target_mask = torch.ones((len(planets),), dtype=torch.bool)
        planet_ids = torch.tensor([_planet_id(planet) for planet in planets], dtype=torch.long)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_attr=global_attr,
            owner_mask=owner_mask,
            source_mask=source_mask,
            target_mask=target_mask,
            planet_ids=planet_ids,
            num_nodes=len(planets),
        )

    def _node_features(self, planet: list[Any], player_id: int) -> list[float]:
        cfg = self.config
        owner = _planet_owner(planet)
        x = _planet_x(planet)
        y = _planet_y(planet)
        dx = x - cfg.center_x
        dy = y - cfg.center_y
        orbit_radius = math.hypot(dx, dy)
        orbit_angle = math.atan2(dy, dx)
        return [
            1.0 if owner == player_id else 0.0,
            1.0 if owner not in {-1, player_id} else 0.0,
            1.0 if owner == -1 else 0.0,
            x / cfg.board_size,
            y / cfg.board_size,
            _planet_radius(planet) / cfg.max_radius,
            min(_planet_ships(planet), cfg.max_ships) / cfg.max_ships,
            _planet_production(planet) / cfg.max_production,
            1.0 if self._is_rotating(planet) else 0.0,
            orbit_radius / cfg.board_size,
            math.sin(orbit_angle),
            math.cos(orbit_angle),
        ]

    def _edges(self, planets: list[list[Any]], player_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        if len(planets) <= 1:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, self.edge_feature_dim), dtype=torch.float32),
            )

        indices: list[tuple[int, int]] = []
        attrs: list[list[float]] = []
        for src_idx, src in enumerate(planets):
            for tgt_idx, tgt in enumerate(planets):
                if src_idx == tgt_idx:
                    continue
                indices.append((src_idx, tgt_idx))
                attrs.append(self._edge_features(src, tgt, player_id))

        edge_index = torch.tensor(indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attrs, dtype=torch.float32)
        return edge_index, edge_attr

    def _edge_features(self, src: list[Any], tgt: list[Any], player_id: int) -> list[float]:
        cfg = self.config
        dx = _planet_x(tgt) - _planet_x(src)
        dy = _planet_y(tgt) - _planet_y(src)
        dist = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        src_owner = _planet_owner(src)
        tgt_owner = _planet_owner(tgt)
        return [
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
            1.0 if self._shot_crosses_sun(src, tgt) else 0.0,
            min(_planet_ships(src), cfg.max_ships) / cfg.max_ships,
            min(_planet_ships(tgt), cfg.max_ships) / cfg.max_ships,
        ]

    def _global_features(
        self,
        raw_obs: Any,
        planets: list[list[Any]],
        fleets: list[list[Any]],
        player_id: int,
    ) -> list[float]:
        cfg = self.config
        my_planets = [planet for planet in planets if _planet_owner(planet) == player_id]
        enemy_planets = [planet for planet in planets if _planet_owner(planet) not in {-1, player_id}]
        neutral_planets = [planet for planet in planets if _planet_owner(planet) == -1]
        my_fleets = [fleet for fleet in fleets if _fleet_owner(fleet) == player_id]
        enemy_fleets = [fleet for fleet in fleets if _fleet_owner(fleet) != player_id]
        ship_denom = max(cfg.max_planets * cfg.max_ships, 1.0)
        return [
            float(_obs_get(raw_obs, "step", 0)) / cfg.episode_steps,
            float(_obs_get(raw_obs, "angular_velocity", 0.0)),
            len(planets) / cfg.max_planets,
            len(my_planets) / cfg.max_planets,
            len(enemy_planets) / cfg.max_planets,
            len(neutral_planets) / cfg.max_planets,
            sum(_planet_ships(planet) for planet in my_planets) / ship_denom,
            sum(_planet_ships(planet) for planet in enemy_planets) / ship_denom,
            sum(_planet_ships(planet) for planet in neutral_planets) / ship_denom,
            sum(_fleet_ships(fleet) for fleet in my_fleets) / ship_denom,
            sum(_fleet_ships(fleet) for fleet in enemy_fleets) / ship_denom,
            len(my_fleets) / max(cfg.max_planets, 1),
        ]

    def _is_rotating(self, planet: list[Any]) -> bool:
        cfg = self.config
        radius = math.hypot(_planet_x(planet) - cfg.center_x, _planet_y(planet) - cfg.center_y)
        return radius + _planet_radius(planet) < cfg.rotation_radius_limit

    def _shot_crosses_sun(self, src: list[Any], tgt: list[Any]) -> bool:
        cfg = self.config
        return (
            _point_to_segment_distance(
                (cfg.center_x, cfg.center_y),
                (_planet_x(src), _planet_y(src)),
                (_planet_x(tgt), _planet_y(tgt)),
            )
            < cfg.sun_radius
        )
    

class OrbitWarsReplayDataset(Dataset):
    """
    Imitation-learning dataset for pointer-policy GNNs.

    Each sample is a PyG Data object with graph tensors and labels:
      y_source: local node index of the launching planet
      y_target: local node index inferred from action_angle
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

        data = self.graph_builder.obs_to_data(raw_obs, player_id)

        data.y_source = torch.tensor([labels["source_idx"]], dtype=torch.long)
        data.y_target = torch.tensor([labels["target_idx"]], dtype=torch.long)
        data.y_ship_pct = torch.tensor([labels["ship_pct"]], dtype=torch.float32)

        data.action_source_id = torch.tensor([labels["source_id"]], dtype=torch.long)
        data.action_target_id = torch.tensor([labels["target_id"]], dtype=torch.long)

        data.action_angle = torch.tensor([row.action_angle], dtype=torch.float32)
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
                    "target_idx": IGNORE_INDEX,
                    "target_id": -1,
                    "ship_pct": 0.0,
                    "source_id": -1,
                }

            samples.append((i, labels))

        return samples

    def _labels_for_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        raw_obs = row.raw_obs
        planets = _sorted_planets(raw_obs)
        id_to_idx = {_planet_id(planet): idx for idx, planet in enumerate(planets)}
        if not planets:
            return None

        if pd.isna(row.action_source):
            return None
        
        source_id = int(row.action_source)
        if source_id not in id_to_idx:
            return None

        source_idx = id_to_idx[source_id]
        source_planet = planets[source_idx]
        action_angle = float(row.action_angle)
        action_ships = float(row.action_ships)
        observed_ships = float(_planet_ships(source_planet))
        available_ships = observed_ships + action_ships if self.assume_post_action_obs else observed_ships
        available_ships = max(available_ships, 1.0)
        ship_pct = float(np.clip(action_ships / available_ships, 0.0, 1.0))

        target_idx = infer_target_index_from_angle(
            raw_obs,
            source_id=source_id,
            action_angle=action_angle,
            action_ships=action_ships,
            config=self.graph_builder.config,
        )
        target_id = _planet_id(planets[target_idx]) if target_idx != IGNORE_INDEX else -1

        if target_idx == IGNORE_INDEX and self.skip_invalid:
            return None

        return {
            "source_id": source_id,
            "source_idx": source_idx,
            "target_id": int(target_id) if target_id is not None else -1,
            "target_idx": target_idx,
            "ship_pct": ship_pct,
        }


def infer_target_index_from_angle(
    raw_obs: Any,
    *,
    source_id: int,
    action_angle: float,
    action_ships: float | None = None,
    config: GraphFeatureConfig | None = None,
) -> int:
    """Infer the target node by simulating the launched fleet's first contact."""
    cfg = config or GraphFeatureConfig()
    planets = _sorted_planets(raw_obs)
    id_to_idx = {_planet_id(planet): idx for idx, planet in enumerate(planets)}
    if source_id not in id_to_idx or len(planets) <= 1:
        return IGNORE_INDEX

    source = planets[id_to_idx[source_id]]
    ships = float(action_ships) if action_ships is not None else _planet_ships(source)
    if not math.isfinite(action_angle) or not math.isfinite(ships) or ships <= 0.0:
        return IGNORE_INDEX

    speed = _fleet_speed(ships, cfg)
    dir_x = math.cos(action_angle)
    dir_y = math.sin(action_angle)
    start_x = _planet_x(source) + dir_x * (_planet_radius(source) + cfg.launch_clearance)
    start_y = _planet_y(source) + dir_y * (_planet_radius(source) + cfg.launch_clearance)

    initial_by_id = {_planet_id(planet): planet for planet in _initial_planets(raw_obs, planets)}
    angular_velocity = float(_obs_get(raw_obs, "angular_velocity", 0.0))
    comets = _obs_get(raw_obs, "comets", [])
    comet_ids = {int(planet_id) for planet_id in _obs_get(raw_obs, "comet_planet_ids", [])}
    horizon = _target_inference_horizon(raw_obs, cfg)
    timestep = max(0.05, min(1.0, float(cfg.target_inference_timestep)))

    sun_distance = _ray_circle_entry_distance(
        start_x,
        start_y,
        dir_x,
        dir_y,
        cfg.center_x,
        cfg.center_y,
        cfg.sun_radius,
    )
    sun_hit_time = sun_distance / speed if sun_distance is not None else None

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
        for idx, planet in enumerate(planets):
            if _planet_id(planet) == source_id:
                continue

            future_position = _predict_planet_position(
                planet,
                next_elapsed,
                cfg,
                initial_by_id,
                angular_velocity,
                comets,
                comet_ids,
            )
            if future_position is None:
                continue

            hit_fraction = _segment_circle_entry_fraction(
                segment_start,
                segment_end,
                future_position,
                _planet_radius(planet),
            )
            if hit_fraction is None:
                continue

            hit_time = elapsed + (next_elapsed - elapsed) * hit_fraction
            hit_distance = speed * hit_time
            score = (hit_time, hit_distance, idx)
            if step_best is None or score < step_best:
                step_best = score

        if step_best is not None:
            hit_time, _, target_idx = step_best
            if sun_hit_time is not None and sun_hit_time <= hit_time + 1e-9:
                return IGNORE_INDEX
            return target_idx

        if sun_hit_time is not None and sun_hit_time <= next_elapsed + 1e-9:
            return IGNORE_INDEX

        elapsed = next_elapsed

    return IGNORE_INDEX


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


def _player_id(raw_obs: Any, fallback: int | None = None) -> int:
    value = _obs_get(raw_obs, "player", fallback if fallback is not None else 0)
    return int(value)


def _sorted_planets(raw_obs: Any) -> list[list[Any]]:
    return sorted((list(row) for row in _obs_get(raw_obs, "planets", [])), key=lambda row: int(row[0]))


def _fleets(raw_obs: Any) -> list[list[Any]]:
    return [list(row) for row in _obs_get(raw_obs, "fleets", [])]


def _initial_planets(raw_obs: Any, fallback: list[list[Any]]) -> list[list[Any]]:
    rows = _obs_get(raw_obs, "initial_planets", [])
    if rows is None or len(rows) == 0:
        rows = fallback
    return sorted((list(row) for row in rows), key=lambda row: int(row[0]))


def _planet_id(planet: list[Any]) -> int:
    return int(planet[0])


def _planet_owner(planet: list[Any]) -> int:
    return int(planet[1])


def _planet_x(planet: list[Any]) -> float:
    return float(planet[2])


def _planet_y(planet: list[Any]) -> float:
    return float(planet[3])


def _planet_radius(planet: list[Any]) -> float:
    return float(planet[4])


def _planet_ships(planet: list[Any]) -> float:
    return float(planet[5])


def _planet_production(planet: list[Any]) -> float:
    return float(planet[6])


def _fleet_owner(fleet: list[Any]) -> int:
    return int(fleet[1])


def _fleet_ships(fleet: list[Any]) -> float:
    return float(fleet[6])


def _fleet_speed(ships: float, cfg: GraphFeatureConfig) -> float:
    ships = max(1.0, float(ships))
    if ships <= 1.0:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (max(1.0, cfg.max_fleet_speed) - 1.0) * (ratio**1.5)


def _target_inference_horizon(raw_obs: Any, cfg: GraphFeatureConfig) -> float:
    configured_horizon = max(1.0, float(cfg.target_inference_horizon))
    current_step = float(_obs_get(raw_obs, "step", 0))
    remaining_steps = float(cfg.episode_steps) - current_step
    if remaining_steps <= 0.0:
        return configured_horizon
    return min(configured_horizon, max(1.0, remaining_steps))


def _predict_planet_position(
    planet: list[Any],
    turns: float,
    cfg: GraphFeatureConfig,
    initial_by_id: dict[int, list[Any]],
    angular_velocity: float,
    comets: list[Any],
    comet_ids: set[int],
) -> tuple[float, float] | None:
    planet_id = _planet_id(planet)
    if planet_id in comet_ids:
        return _predict_comet_position(planet_id, comets, turns)

    initial = initial_by_id.get(planet_id, planet)
    orbit_radius = math.hypot(_planet_x(initial) - cfg.center_x, _planet_y(initial) - cfg.center_y)
    if orbit_radius + _planet_radius(initial) >= cfg.rotation_radius_limit or angular_velocity == 0.0:
        return _planet_x(planet), _planet_y(planet)

    current_angle = math.atan2(_planet_y(planet) - cfg.center_y, _planet_x(planet) - cfg.center_x)
    future_angle = current_angle + angular_velocity * turns
    return (
        cfg.center_x + orbit_radius * math.cos(future_angle),
        cfg.center_y + orbit_radius * math.sin(future_angle),
    )


def _predict_comet_position(
    planet_id: int,
    comets: list[Any],
    turns: float,
) -> tuple[float, float] | None:
    for group in comets:
        planet_ids = group.get("planet_ids", [])
        if planet_id not in planet_ids:
            continue

        comet_idx = np.where(planet_ids == planet_id)[0]
        comet_idx = int(comet_idx[0])

        paths = group.get("paths", [])
        path_index = int(group.get("path_index", 0))
        if comet_idx >= len(paths):
            return None

        path = paths[comet_idx]
        future_index = path_index + max(0.0, turns)
        lower_idx = int(math.floor(future_index))
        upper_idx = int(math.ceil(future_index))
        if lower_idx < 0 or lower_idx >= len(path):
            return None
        if upper_idx >= len(path):
            if upper_idx == lower_idx:
                return float(path[lower_idx][0]), float(path[lower_idx][1])
            return None

        lower = path[lower_idx]
        upper = path[upper_idx]
        fraction = future_index - lower_idx
        return (
            float(lower[0]) + (float(upper[0]) - float(lower[0])) * fraction,
            float(lower[1]) + (float(upper[1]) - float(lower[1])) * fraction,
        )
    return None


def _ray_circle_entry_distance(
    ray_x: float,
    ray_y: float,
    dir_x: float,
    dir_y: float,
    circle_x: float,
    circle_y: float,
    radius: float,
) -> float | None:
    rel_x = ray_x - circle_x
    rel_y = ray_y - circle_y
    c = rel_x * rel_x + rel_y * rel_y - radius * radius
    if c <= 0.0:
        return 0.0

    b = 2.0 * (rel_x * dir_x + rel_y * dir_y)
    discriminant = b * b - 4.0 * c
    if discriminant < 0.0:
        return None

    sqrt_discriminant = math.sqrt(discriminant)
    first = (-b - sqrt_discriminant) / 2.0
    second = (-b + sqrt_discriminant) / 2.0
    if second < 0.0:
        return None
    return max(0.0, first)


def _segment_circle_entry_fraction(
    start: tuple[float, float],
    end: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> float | None:
    seg_x = end[0] - start[0]
    seg_y = end[1] - start[1]
    a = seg_x * seg_x + seg_y * seg_y
    rel_x = start[0] - center[0]
    rel_y = start[1] - center[1]
    c = rel_x * rel_x + rel_y * rel_y - radius * radius
    if c <= 0.0:
        return 0.0
    if a <= 1e-12:
        return None

    b = 2.0 * (rel_x * seg_x + rel_y * seg_y)
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None

    sqrt_discriminant = math.sqrt(discriminant)
    first = (-b - sqrt_discriminant) / (2.0 * a)
    second = (-b + sqrt_discriminant) / (2.0 * a)
    if 0.0 <= first <= 1.0:
        return first
    if 0.0 <= second <= 1.0:
        return second
    return None


def _angle_delta(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


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
