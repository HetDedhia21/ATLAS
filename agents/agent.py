import random

class QAgent:
    def __init__(self, actions):
        self.q_table = {}  # (state, action) → value
        self.actions = actions

        # hyperparameters
        self.alpha = 0.1          # learning rate
        self.gamma = 0.9          # discount factor
        self.epsilon = 1.0        # start high
        self.epsilon_min = 0.05   # don't go below this
        self.epsilon_decay = 0.96

    def get_state_key(self, state):
        return tuple(state)  # make it hashable

    def choose_action(self, state):
        state_key = self.get_state_key(state)

        # exploration
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # exploitation
        q_values = [self.q_table.get((state_key, a), 0) for a in self.actions]
        max_q = max(q_values)

        # pick randomly among best actions
        best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        current_q = self.q_table.get((state_key, action), 0)

        next_qs = [self.q_table.get((next_key, a), 0) for a in self.actions]
        max_next_q = max(next_qs)

        new_q = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )

        self.q_table[(state_key, action)] = new_q