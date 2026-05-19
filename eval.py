from kaggle_environments import make
import torch
from agent.base import NearestPlanetAgent
from agent.kaggle_agent import PPOKaggleAgent

if __name__ == "__main__":
    CHECKPOINT_PATH = "/home/tts26/sonh/Orbit-Wars-RL/orbit_wars_checkpoint/ActorCritic/20260519_033439/latest_ckpt.pth" # Point to your saved model
    
    print("Loading agents...")
    ppo_agent = PPOKaggleAgent(CHECKPOINT_PATH, device="cuda" if torch.cuda.is_available() else "cpu")
    nn_agent = NearestPlanetAgent()

    print("Initializing Orbit Wars environment...")
    env = make("orbit_wars", configuration={"seed": 5}, debug=False)

    print("Running Match: PPO (Player 1) vs NearestNeighbor (Player 2)...")
    # env.run takes a list of callables.
    # We pass the bound .act methods of both classes!
    env.run([ppo_agent.act, nn_agent.act])

    print("Match Complete! Final Status:")
    final_state = env.steps[-1]
    for i, s in enumerate(final_state):
        print(f"Player {i}: reward={s.reward}, status={s.status}")

    print("Rendering HTML...")
    html_output = env.render(mode="html", width=800, height=600)
    with open("orbit_wars_render.html", "w") as f:
        f.write(html_output)
    print("Saved replay to orbit_wars_render.html")