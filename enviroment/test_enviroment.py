import pprint
from pettingzoo.test import parallel_api_test

# Import your environment and dependencies here
# from your_module import OrbitWarsMARLWrapper, Config, ObsProcessor, ActProcessor
from enviroment.orbit_wars import OrbitWarsWrapper
from enviroment.processor import PaddedObservationProcessor, FixedActionProcessor

class Config:
    def __init__(self):
        self.env = self.EnvConfig()

    class EnvConfig:
        def __init__(self):
            self.max_planets = 10
            self.candidate_count = 5
            self.board_size = 100.0
            self.episode_steps = 500
            self.candidate_count = 8
            self.ship_bucket_count = 8
            self.max_planets = 48
            self.max_ships = 400.0
            self.max_production = 5.0

def test_orbit_wars_env():
    print("=== Setting up the Environment ===")
    # 1. Initialize your specific config and processors (Replace with your actual classes)
    config = Config()
    obs_processor = PaddedObservationProcessor()
    act_processor = FixedActionProcessor()
    
    env = OrbitWarsWrapper(config, obs_processor, act_processor)
    
    # --- For the sake of this script, assuming 'env' is instantiated ---
    
    print("\n=== Running PettingZoo API Test ===")
    # This will throw an error if your env breaks any standard PettingZoo rules
    try:
        parallel_api_test(env, num_cycles=100)
        print("API Test Passed! Your environment is perfectly compliant.")
    except Exception as e:
        print(f"API Test Failed. See error: {e}")
        return

    print("\n=== Starting Step-by-Step Execution ===")
    
    # 2. Reset the environment
    obs, infos = env.reset()
    
    print("\n--- STEP 0 (Reset) ---")
    print(f"Active Agents: {env.agents}")
    print("Initial Observations:")
    pprint.pprint(obs, depth=2) # depth=2 keeps giant arrays from flooding the console
    print("Initial Infos:")
    pprint.pprint(infos)

    # 3. Run a test loop for a fixed number of steps
    max_steps = 0
    step_count = 0
    
    # Run while there are still active agents and we haven't hit our step limit
    while env.agents and step_count < max_steps:
        step_count += 1
        print(f"\n--- STEP {step_count} ---")
        
        # 4. Sample random actions for all currently active agents
        actions = {}
        for agent in env.agents:
            # Assumes your act_processor.get_space() returned a valid gymnasium Space
            actions[agent] = env.action_space(agent).sample() 
            
        print("Sampled Actions:")
        pprint.pprint(actions)
        
        # 5. Step the environment
        obs, rewards, terminations, truncations, infos = env.step(actions)
        pprint.pprint("Observations:")
        pprint.pprint(obs, depth=2)
        
        # 6. Print the results of the step
        print("\nRewards:")
        pprint.pprint(rewards)
        
        print("\nTerminations (Done):")
        pprint.pprint(terminations)
        
        print("\nTruncations (Timeouts/Errors):")
        pprint.pprint(truncations)
        
        print("\nInfos (Status):")
        pprint.pprint(infos)
        
        # Optional: Print observation keys or shapes to avoid console spam
        print("\nObservation Keys:")
        for agent, agent_obs in obs.items():
            print(f"  {agent}: {list(agent_obs.keys()) if isinstance(agent_obs, dict) else 'Array shape: ' + str(getattr(agent_obs, 'shape', 'Unknown'))}")
            
    print(f"\n=== Test Finished after {step_count} steps ===")

if __name__ == "__main__":
    # Uncomment and run once you have your 'env' instantiated inside the function
    test_orbit_wars_env()