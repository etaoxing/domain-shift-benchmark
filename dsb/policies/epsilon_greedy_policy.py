from dsb.dependencies import *

from .base_policy import BasePolicy


class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, agent, epsilon=None, **kwargs):
        super().__init__(agent, **kwargs)
        self._epsilon = epsilon  # if epsilon is None, then check agent.epsilon

    def select_action(self, state, deterministic=False, **kwargs):
        state, action = self.agent.select_action(state, deterministic=deterministic, **kwargs)
        if not deterministic:
            eps = self.agent.epsilon if self._epsilon is None else self._epsilon
            mask = np.rng.random(action.shape[0]) < eps
            rnd_indices = np.nonzero(mask)[0]  # get 0th dim
            if len(rnd_indices) > 0:
                action[rnd_indices, ...] = [
                    self.agent.mdp_space.action_space.sample() for _ in range(len(rnd_indices))
                ]
        return state, action
