import random
from simulations.multi_env import MultiAgentEnv

env = MultiAgentEnv()
states = env.reset()

for _ in range(5):
    actions = {j: random.choice([0, 1]) for j in states}
    states, rewards = env.step(actions)

    print("States:", states)
    print("Rewards:", rewards)
    print("------")