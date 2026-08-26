"""
ATLAS - Stage 4: Per-agent Q-network for QMIX (paper Section IV-B).

Each junction agent keeps a small feedforward MLP mapping its local
observation (plus incoming neighbor messages, Section IV-C) to one
Q-value per action. Feedforward, not recurrent - the paper's rationale
(Sec IV-B) is that the observation already carries what's needed for
the decision, no temporal memory required.
"""

import random
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """obs_dim -> hidden -> hidden -> num_actions"""

    def __init__(self, obs_dim, num_actions, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class QMixAgent:
    """
    Wraps a QNetwork with epsilon-greedy action selection, mirroring the
    interface of agents.agent.QAgent (choose_action / epsilon fields) so
    Stage 4's training loop reads the same as Stage 3's, but backed by a
    neural net instead of a tabular q_table, and trained jointly through
    the mixing network rather than independently.
    """

    def __init__(self, obs_dim, num_actions, hidden_dim=64, device="cpu"):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.device = device

        self.q_network = QNetwork(obs_dim, num_actions, hidden_dim).to(device)
        self.target_network = QNetwork(obs_dim, num_actions, hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # same schedule as Stage 3's QAgent, for a fair comparison
        self.epsilon = 1.0
        self.epsilon_min = 0.15
        self.epsilon_decay = 0.97

    def choose_action(self, obs_vector):
        """
        Returns (action, was_greedy). was_greedy=True means the action came
        from argmax over the Q-network (exploit); False means it was a
        random exploration pick. Exposed so the training loop can log the
        exploit-only action distribution per junction - if that distribution
        collapses onto a single action regardless of state, the Q-network
        has likely gone degenerate rather than learning a real policy.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions), False

        with torch.no_grad():
            obs_t = torch.as_tensor(obs_vector, dtype=torch.float32, device=self.device)
            q_values = self.q_network(obs_t.unsqueeze(0))
            return int(torch.argmax(q_values, dim=1).item()), True

    def update_target(self):
        """Hard update (paper Sec IV-F step 7)."""
        self.target_network.load_state_dict(self.q_network.state_dict())