"""
ATLAS - Stage 4: Joint experience replay buffer (paper Section IV-E).

Stores transitions (s, o, a, r, s', o') across all five agents together,
since QMIX trains on the *joint* action-value estimate Q_tot rather than
per-agent rewards in isolation - unlike Stage 3, where each agent's
QAgent.update() only ever sees its own (state, action, reward, next_state).
"""

import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, obs, actions, reward, next_state, next_obs):
        """
        state / next_state : global state vector s / s'   (Sec III-E)
        obs / next_obs      : dict {junction_id: obs_vector} o / o' (Sec III-F)
        actions              : dict {junction_id: action_index}
        reward                : scalar team reward r (sum of per-agent rewards)
        """
        self.buffer.append((state, obs, actions, reward, next_state, next_obs))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
