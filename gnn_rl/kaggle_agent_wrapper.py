import torch
from typing import Any, Dict, List
from datasets import OrbitWarsGraphBuilder, _obs_get
from torch_geometric.data import Batch

from models import GNNAgent


class KaggleGNNWrapper:
    def __init__(self, model: torch.nn.Module, device: torch.device, graph_builder: OrbitWarsGraphBuilder) -> None:
        self.model = model
        self.device = device
        self.graph_builder = graph_builder
        self.model.to(self.device)
        self.model.eval() 
        
        self.max_planets = getattr(self.model, "max_planets", getattr(self.model.network, "max_planets", 60))

        self.hx = None
        self.cx = None
        
    def reset(self) -> None:
        """Reset trạng thái khi bắt đầu game mới."""
        self.hx = None
        self.cx = None

    def _extract_features_from_obs(self, obs: Any, config: Any) -> Any:
        my_player = obs.player if hasattr(obs, 'player') else obs.get("player", 0)
        data = self.graph_builder.obs_to_data(obs, my_player)
        return data

    def act(self, obs: Any = None, config: Any = None) -> List[Dict[str, Any]]:
        with torch.no_grad():
            if obs and hasattr(obs, 'step') and obs.step == 0:
                self.reset()
            elif isinstance(obs, dict) and obs.get("step", -1) == 0:
                self.reset()

            data = self._extract_features_from_obs(obs, config)
            batch = Batch.from_data_list([data]).to(self.device)

            outputs, _, _, _ = self.model.get_action_and_value(batch, deterministic=False)
            
            action_cpu = outputs.squeeze(0).cpu().numpy() 
            
            target_actions = action_cpu[:, 0]
            ship_pcts = action_cpu[:, 1]
            planet_ids = data.planet_ids.cpu().numpy()

            kaggle_actions = []

            planets = obs.planets if hasattr(obs, 'planets') else obs.get("planets", [])
            planet_ships = {int(p[0]): int(p[5]) for p in planets}

            max_idx = min(len(target_actions), len(planet_ids))

            for i in range(max_idx):
                source_planet_id = int(planet_ids[i])
                
                if source_planet_id == -1: 
                    continue
                
                # Kiểm tra nếu Agent chọn "NO_ACTION"
                target_idx = int(target_actions[i])
                if target_idx == self.max_planets: 
                    continue
                    
                if target_idx < 0 or target_idx >= len(planet_ids):
                    continue
                    
                target_planet_id = int(planet_ids[target_idx])
                
                if target_planet_id == -1 or target_planet_id == source_planet_id:
                    continue

                current_ships = planet_ships.get(source_planet_id, 0)
                if current_ships == 0: 
                    continue
                
                num_ship_buckets = getattr(self.model, "num_ship_buckets", getattr(self.model.network, "num_ship_buckets", 20))
                pct = float(ship_pcts[i]) / (num_ship_buckets - 1)
                
                num_ships = int(pct * current_ships)
                num_ships = max(1, num_ships)
                if num_ships > current_ships: 
                    num_ships = current_ships
                
                # TÍNH GÓC BẮN THÔNG QUA WORLD_MODEL
                shot_result = self.graph_builder.world_model.plan_shot(source_planet_id, target_planet_id, num_ships)
                
                if shot_result is None:
                    continue 
                
                angle = float(shot_result[0])
                
                command = [source_planet_id, angle, num_ships]
                kaggle_actions.append(command)

            return kaggle_actions