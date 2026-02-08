#rl_buffer.py
import numpy as np


class RolloutBuffer(object):
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.costs = []
        self.dones = []

    def add(self, obs, action, logprob, value, reward, cost, done):
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.logprobs.append(float(logprob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.costs.append(float(cost))
        self.dones.append(bool(done))

    def size(self):
        return len(self.rewards)

    def is_full(self):
        return self.size() >= self.capacity

    def mark_last_done(self, reward_bonus=0.0, cost_override=None):
        if not self.rewards:
            return
        self.rewards[-1] += float(reward_bonus)
        self.dones[-1] = True
        if cost_override is not None:
            self.costs[-1] = float(cost_override)
