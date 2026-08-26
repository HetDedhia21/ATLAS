"""
ATLAS - Stage 4: QMIX mixing network (paper Section IV-D).

Combines the five per-agent Q-values into a single joint estimate
Q_tot, conditioned on the global state s. Hypernetworks map s to the
mixing network's weights and biases; weights are passed through
torch.abs() to enforce the monotonicity constraint dQ_tot/dQ_i >= 0
for every agent, which is exactly what makes decentralized execution
valid despite centralized training (Sec IV-D / IV-E).

Training-only: at execution/deployment this whole module is discarded
and each agent just argmaxes its own QNetwork (Sec IV-E).
"""

import torch
import torch.nn as nn


class MixingNetwork(nn.Module):
    def __init__(self, num_agents, state_dim, embed_dim=32, hyper_hidden_dim=64):
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim

        # hyper_w1: state -> (num_agents x embed_dim) weight matrix
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, num_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # hyper_w2: state -> (embed_dim x 1) weight vector
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs, state):
        """
        agent_qs: (batch, num_agents) - each agent's chosen-action Q-value
        state:    (batch, state_dim)  - global state s (Sec III-E / IV-D)
        returns:  (batch, 1) Q_tot
        """
        batch_size = agent_qs.size(0)
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)

        # layer 1 - non-negative weights enforce monotonicity
        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)
        hidden = torch.nn.functional.elu(torch.bmm(agent_qs, w1) + b1)

        # layer 2 - non-negative weights, same reason
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.view(batch_size, 1)
