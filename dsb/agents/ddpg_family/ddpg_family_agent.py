from dsb.dependencies import *
import dsb.builder as builder
from dsb.utils import torchify, untorchify

from ..utils import update_target_network
from ..base_agent import BaseAgent

from .ddpg import DDPG
from .uvf_ddpg import UVFDDPG
from .td3 import TD3
from .sac import SAC

__LearnerCls__ = builder.registry([DDPG, UVFDDPG, TD3, SAC])


class DDPGFamilyAgent(BaseAgent):
    def __init__(
        self,
        mdp_space,
        learner_params=dict(cls='DDPG'),
        use_target_embedding_head=False,
        use_target_for_relabeling=False,
        embedding_head_tau=None,
        learner_optimize_embedding_head=True,
        learner_optimize_embedding_head_params=dict(),
        **kwargs,
    ):
        super().__init__(mdp_space, **kwargs)
        self.use_target_embedding_head = use_target_embedding_head
        self.use_target_for_relabeling = use_target_for_relabeling
        self.embedding_head_tau = embedding_head_tau

        self.learner_optimize_embedding_head = learner_optimize_embedding_head

        if self.embedding_head and self.learner_optimize_embedding_head:
            embedding_head_param_group = {
                'params': self.embedding_head.parameters(),
                **learner_optimize_embedding_head_params,
            }
        else:
            embedding_head_param_group = None

        self.learner = builder.build_module(
            __LearnerCls__,
            learner_params,
            mdp_space,
            embedding_head_param_group=embedding_head_param_group,
        )

        if self.embedding_head and self.use_target_embedding_head:
            # see https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/sac_ae.py#L410
            self.embedding_head_target = copy.deepcopy(self.embedding_head)
            self.embedding_head_target.load_state_dict(self.embedding_head.state_dict())

    def select_action(self, state, deterministic=None, state_embedding=None, **kwargs):
        with torch.no_grad():
            x = torchify(state)

            if self.embedding_head:
                # TODO: substitute in state_embeddings
                x = self.compute_embedding(x)
            else:
                if self.state_normalizer:
                    x = self.state_normalizer(x)

            action = self.learner.select_action(x, deterministic=deterministic)
            return state, untorchify(action)

    def optimize(self, batch_it, batch_size, replay_buffer):
        opt_info = {}

        ptrs, batch = replay_buffer.sample(batch_it, batch_size)
        batch_dtype = None if replay_buffer.batch_dtype is not None else torch.float32
        batch = tuple(torchify(x, dtype=batch_dtype) for x in batch)
        _state, action, _next_state, reward, done = batch

        # optimize auxilary first on batch
        # NOTE: this order is different than SAC+AE or SODA which does auxilary updates after RL
        if self.embedding_head and getattr(self.embedding_head, 'optimize_interval', False):
            embedding_opt_info = self.optimize_embedding_head(replay_buffer, ptrs, batch)
            opt_info.update(embedding_opt_info)

        # recompute b/c auxilary losses may have updated embedding
        # so the compute graph is freed (otherwise would need to set retain_graph=True)
        if self.embedding_head:
            state = self.compute_embedding(_state)
            next_state = self.compute_embedding(
                _next_state, use_target_embedding_head=self.use_target_embedding_head
            )
        else:
            if self.state_normalizer:
                state = self.state_normalizer(_state)
                next_state = self.state_normalizer(_next_state)  # TODO: target state normalizer?
            else:
                state, next_state = _state, _next_state

        # goal reached relabeling
        if '_goal_relabeled_mask' in next_state.keys():
            next_state_embedding = (
                None
                if (self.use_target_embedding_head and not self.use_target_for_relabeling)
                else next_state
            )

            reward, done = relabel_goal_for_transition(
                replay_buffer, _next_state, reward, done, next_state_embedding=next_state_embedding
            )

        # RL updates
        rl_batch = (state, action, next_state, reward, done)
        learner_opt_info = self.learner.optimize(self.optimize_iterations, rl_batch)
        if self.optimize_iterations % self.learner.targets_update_interval == 0:
            if self.embedding_head and self.use_target_embedding_head:
                update_target_network(
                    self.embedding_head,
                    self.embedding_head_target,
                    tau=(
                        self.embedding_head_tau
                        if self.embedding_head_tau is not None
                        else self.learner.tau
                    ),
                )
        opt_info.update(learner_opt_info)

        return opt_info
