import torch
from datasets import OrbitWarsGraphBuilder, _obs_get, Planet
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
            node_dim=21,     
            edge_dim=18,     
            global_dim=21,   
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)
        
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
        player_id = _obs_get(obs, "player", 0)
        actions = []
        
        # Khởi tạo các trạng thái "bóng" (Shadow States) cho lượt này
        simulated_fleets = []
        deductions = {}
        
        # Khởi tạo bộ nhớ LSTM (Trống vào đầu mỗi Turn mới)
        hx, cx = None, None
        
        # Mapping để dễ dàng tra cứu lượng quân gốc
        planets_raw = _obs_get(obs, "planets", [])
        planets = [Planet(*planet_raw) for planet_raw in planets_raw]
        base_ships_map = {p.id: p.ships for p in planets}

        # Giới hạn số hành động mỗi lượt để chống infinite loop (Safe-guard)
        MAX_ACTIONS_PER_TURN = 10 

        for _ in range(MAX_ACTIONS_PER_TURN):
            # 1. Trích xuất đồ thị VỚI các hành động nháp đã thực hiện
            data = self.graph_builder.obs_to_data(
                obs, 
                player_id, 
                deductions=deductions, 
                simulated_fleets=simulated_fleets
            ).to(self.device)

            # 2. Dự đoán hành động tiếp theo
            with torch.no_grad():
                output = self.model.act(data, hx=hx, cx=cx, deterministic=True)
            
            # Cập nhật bộ nhớ cho vòng lặp tiếp theo
            hx, cx = output["hx"], output["cx"]
            
            # 3. Trích xuất kết quả
            source_idx = output["source"].item()
            planet_id = data.planet_ids[source_idx].item()

            # 4. KIỂM TRA ĐIỀU KIỆN DỪNG (END TURN)
            # Dummy node luôn được gán ID là -1 trong GraphBuilder
            if planet_id == -1:
                break  # Mô hình quyết định dừng xuất quân

            angle = output["angle"].item()
            ship_pct = output["ship_pct"].item()

            # 5. Tính toán số quân xuất kích
            # Lấy số quân GỐC trừ đi số quân đã sử dụng ở các bước trước trong cùng turn
            base_ships = float(base_ships_map.get(planet_id, 0.0))
            available_ships = max(0.0, base_ships - deductions.get(planet_id, 0.0))
            
            # Tính lượng quân thực tế và làm tròn
            ship_count = int(available_ships * min(max(ship_pct, 0.0), 1.0))
            ship_count = max(1, ship_count) # Kaggle yêu cầu gửi ít nhất 1 quân
            
            # Nếu vì lý do gì đó lượng quân tính ra <= 0 (thường là do làm tròn), bỏ qua hành động này
            if ship_count <= 0 or available_ships <= 0:
                continue

            # 6. Ghi nhận hành động chuẩn của Kaggle
            actions.append([planet_id, float(angle), ship_count])

            # 7. Cập nhật Shadow States để model nhìn thấy cục diện ở vòng lặp sau
            deductions[planet_id] = deductions.get(planet_id, 0.0) + ship_count
            simulated_fleets.append({
                "from_planet_id": planet_id,
                "angle": angle,
                "ships": ship_count,
                "owner": player_id
            })

        return actions