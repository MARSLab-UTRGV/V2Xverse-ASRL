#rl_policy.py
import torch
import torch.nn as nn
from torch.distributions import Normal


class GammaPolicy(nn.Module):
    """Simple MLP Gaussian policy for gamma actions."""

    def __init__(self, obs_dim, action_dim=3, hidden_sizes=(64, 64)):
        super(GammaPolicy, self).__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.actor = nn.Sequential(*layers, nn.Linear(in_dim, action_dim))

        v_layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            v_layers.append(nn.Linear(in_dim, h))
            v_layers.append(nn.Tanh())
            in_dim = h
        self.critic = nn.Sequential(*v_layers, nn.Linear(in_dim, 1))

        self.logstd = nn.Parameter(torch.zeros(1, action_dim))

    def _ensure_batch(self, x):
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x

    def get_value(self, obs):
        obs = self._ensure_batch(obs)
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs, action=None, deterministic=False):
        obs = self._ensure_batch(obs)
        mean = self.actor(obs)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)

        if action is None:
            if deterministic:
                action = mean
            else:
                action = dist.sample()

        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, logprob, entropy, value

    def evaluate_actions(self, obs, action):
        obs = self._ensure_batch(obs)
        mean = self.actor(obs)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return logprob, entropy, value
