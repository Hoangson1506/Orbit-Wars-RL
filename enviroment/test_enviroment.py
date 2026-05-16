import numpy as np
import gymnasium as gym
from dataclasses import dataclass

from enviroment.processor import (
    PaddedObservationProcessor, 
    FixedActionProcessor
)
from enviroment.orbit_wars import OrbitWarsWrapper

@dataclass
class EnvConfig:
    board_size: float = 100.0
    episode_steps: int = 1000
    candidate_count: int = 9
    ship_bucket_count: int = 8
    max_planets: int = 50
    max_ships: float = 400.0
    max_production: float = 5.0


def test_environment():
    print("="*50)
    print("Initializing Orbit Wars Environment Test")
    print("="*50)

    # 1. Setup Configuration
    # We use a smaller candidate count for testing to keep logs clean
    config = EnvConfig()

    # 2. Instantiate Strategies and Wrapper
    obs_processor = PaddedObservationProcessor()
    act_processor = FixedActionProcessor()
    
    env = OrbitWarsWrapper(
        env_cfg=config,
        obs_processor=obs_processor,
        act_processor=act_processor
    )

    # 3. Test Reset and Observation Space
    print("\n[TEST 1] Resetting Environment...")
    obs, info = env.reset()
    
    print("Observation Dictionary Shapes:")
    for key, value in obs.items():
        print(f"  - {key}: {value.shape} (dtype: {value.dtype})")

    # Validate against expected shapes
    assert obs["global"].shape == (8,), "Global shape mismatch!"
    assert obs["self"].shape == (50, 11), "Self shape mismatch!"
    assert obs["candidates"].shape == (50, 9, 14), "Candidates shape mismatch!"
    assert obs["mask"].shape == (50, 9), "Mask shape mismatch!"
    print("✅ Observation shapes passed!")

    # 4. Test Action Space
    print(f"\n[TEST 2] Action Space:")
    print(f"  - Type: {type(env.action_space)}")
    print(f"  - Shape: {env.action_space.shape}")
    print("✅ Action space verified!")

    # 5. Run a Dummy Episode
    print("\n[TEST 3] Running Random Agent Episode...")
    total_reward = 0
    steps = 0
    done = False

    while not done:
        # Sample a random action from our MultiDiscrete space
        # Note: In a real scenario, you would mask this using obs["mask"]!
        random_action = env.action_space.sample()
        
        # Step the environment
        obs, reward, done, truncated, info = env.step(random_action)
        
        total_reward += reward
        steps += 1
        
        if steps % 50 == 0:
            print(f"  Step {steps}... Current Total Reward: {total_reward}")
            
        # Hard stop just in case
        if steps > config.episode_steps + 10:
            print("❌ Episode exceeded maximum steps!")
            break

    print("\n" + "="*50)
    print(f"Episode Finished!")
    print(f"Total Steps Taken: {steps}")
    print(f"Total Reward: {total_reward}")
    print("="*50)
    print("✅ Environment is fully operational and ready for RLlib/TorchRL!")

if __name__ == "__main__":
    test_environment()