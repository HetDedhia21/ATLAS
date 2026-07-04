from agents.agent import QAgent
from simulations.multi_env import MultiAgentEnv

env = MultiAgentEnv()
recent_rewards = []

agents = {
    j: QAgent(actions=[0, 1])  # assuming 2 phases
    for j in env.junctions
}

for episode in range(100):

    env = MultiAgentEnv()   # reset environment

    states = {j: env.junctions[j].get_state() for j in env.junctions}
    total_reward = 0
    
    for step in range(200):

        # choose actions
        actions = {
            j: agents[j].choose_action(states[j])
            for j in agents
        }

        next_states, rewards = env.step(actions)

        # learning update
        for j in agents:
            agents[j].update(
                states[j],
                actions[j],
                rewards[j],
                next_states[j]
            )

        states = next_states
        total_reward += list(rewards.values())[0]

    for agent in agents.values():
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    recent_rewards.append(total_reward)

    # if episode % 10 == 0:
    #     avg = sum(recent_rewards[-10:]) / 10
    #     print(f"Episode {episode}, Avg Reward: {avg}")

    if episode % 10 == 0:
        print(f"Episode {episode} | Total Reward: {round(total_reward,2)} | Epsilon: {agent.epsilon:.4f}")
