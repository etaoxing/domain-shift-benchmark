from dsb.dependencies import *
import dsb.builder as builder
from dsb.utils import torchify, untorchify

from ..base_agent import BaseAgent

from .mse_behavioral_cloning import MSEBehavioralCloning
from .behavioral_cloning import BehavioralCloning

__ImitatorCls__ = builder.registry([MSEBehavioralCloning, BehavioralCloning])


class ImitationAgent(BaseAgent):
    def __init__(
        self,
        mdp_space,
        imitator_params=dict(cls='BehavioralCloning'),
        imitator_optimize_embedding_head=True,
        imitator_optimize_embedding_head_params=dict(),
        **kwargs,
    ):
        super().__init__(mdp_space, **kwargs)
        self.imitator_optimize_embedding_head = imitator_optimize_embedding_head

        if self.embedding_head and self.imitator_optimize_embedding_head:
            embedding_head_param_group = {
                'params': self.embedding_head.parameters(),
                **imitator_optimize_embedding_head_params,
            }
        else:
            embedding_head_param_group = None

        self.imitator = builder.build_module(
            __ImitatorCls__,
            imitator_params,
            mdp_space,
            embedding_head_param_group=embedding_head_param_group,
        )

    def select_action(self, state, deterministic=None, state_embedding=None, **kwargs):
        with torch.no_grad():
            x = torchify(state)

            if self.embedding_head:
                x = self.compute_embedding(x, state_embedding=state_embedding)
            else:
                if self.state_normalizer:
                    x = self.state_normalizer(x)

            action = self.imitator(x, deterministic=deterministic)
            return state, untorchify(action)

    def optimize(self, batch_it, batch_size, buffer):
        opt_info = {}

        ptrs, batch = buffer.sample(batch_it, batch_size)
        batch = tuple(
            torchify(x, dtype=None if buffer.batch_dtype is not None else torch.float32)
            for x in batch
        )
        _state, action, _next_state, reward, done = batch

        if self.embedding_head and getattr(self.embedding_head, 'optimize_interval', False):
            embedding_opt_info = self.optimize_embedding_head(buffer, ptrs, batch)
            opt_info.update(embedding_opt_info)

        if self.dynamics_head and getattr(self.dynamics_head, 'optimize_interval', False):
            dynamics_opt_info = self.optimize_dynamics_head(buffer, ptrs, batch)
            opt_info.update(dynamics_opt_info)

        if self.embedding_head:
            state = self.compute_embedding(_state)
        else:
            if self.state_normalizer:
                state = self.state_normalizer(_state)
            else:
                state = _state

        if self.imitator and getattr(self.imitator, 'optimize_interval', False):
            if self.optimize_iterations % self.imitator.optimize_interval == 0:
                imitator_opt_info = self.imitator.optimize(state, action)
            else:
                imitator_opt_info = {k: None for k in self.imitator.opt_info_keys}
            opt_info.update(imitator_opt_info)

        return opt_info
