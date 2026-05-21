from my_agents.aggressiveNearestAgent import AggressiveNearestAgent

agent = AggressiveNearestAgent()

def play(obs):
    return agent.act(obs)