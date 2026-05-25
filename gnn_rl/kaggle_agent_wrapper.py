import torch
from datasets import _player_id, _sorted_planets, _fleets, _planet_id, _planet_owner, _planet_ships, Data, OrbitWarsGraphBuilder
from env import graph_action_to_move

# Import your trained model architecture
from models import GNNAgent

class KaggleGNNWrapper:
    """
    Wraps the trained PyTorch GNNAgent so it can communicate with the 
    Kaggle orbit_wars environment.
    """
    def __init__(self, model_path: str, device: str = "cpu", hidden_dim = 128, num_layers=3, dropout=0.1):
        self.device = torch.device(device)
        self.graph_builder = OrbitWarsGraphBuilder()
        
        # Initialize the model architecture (ensure dims match your training setup)
        self.model = GNNAgent(
            node_dim=12,
            edge_dim=13,
            global_dim=12,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
        
        # Load the IL trained weights
        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.model.eval() # Crucial for deterministic evaluation

    def act(self, obs, config):
        """
        The method called by Kaggle Environments at every step.
        """
        # 1. Convert raw Kaggle obs to PyTorch Geometric graph
        data = self.graph_builder.obs_to_data(obs, _player_id(obs)).to(self.device)

        print(data.source_mask.sum())

        # 2. Get model prediction (deterministic=True is standard for evaluation)
        with torch.no_grad():
            output = self.model.act(data, deterministic=True)
        
        # 3. Extract predictions
        source_idx = output["source"].item()
        target_idx = output["target"].item()
        ship_pct = output["ship_pct"].item()

        # 4. Format output to Kaggle's expected action format
        # TODO: Adjust this depending on how Orbit Wars expects the action string/dict.
        # Example: "source_planet_id-target_planet_id-ship_amount"
        
        # You will likely need to map your graph node indices back to actual planet IDs
        # and convert ship_pct to an absolute integer based on the source planet's garrison.
        action = graph_action_to_move(
            obs,
            source_idx=source_idx,
            target_idx=target_idx,
            ship_pct=ship_pct,
        )
        print(action)
        
        return action
