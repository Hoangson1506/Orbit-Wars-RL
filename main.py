from kaggle_environments import make
from agents import NearestPlanetAgent

env = make("orbit_wars", configuration={"seed": 42}, debug=False)
agent = NearestPlanetAgent()
env.run([agent.act, "random"])

final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

env.render(mode="html", width=800, height=600)
with open("orbit_wars_render.html", "w") as f:
    f.write(env.render(mode="html", width=800, height=600))