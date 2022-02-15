from dsb.dependencies import *
from dsb.utils import torchify, untorchify

from ..base_agent import BaseAgent


class NoopAgent(BaseAgent):
    def __init__(self, mdp_space, **kwargs):
        super().__init__(mdp_space, **kwargs)

    def select_action(self, state, high_action_mask=None, **kwargs):
        if self.mdp_space.action_type == 'continuous':
            action = np.zeros(shape=(state['_time_step'].shape[0], self.mdp_space.action_dim))
        else:
            raise NotImplementedError

        return state, action

    def optimize(self, batch_it, batch_size, buffer):
        return {}
