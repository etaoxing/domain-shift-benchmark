from dsb.dependencies import *

from .mixins import EmbeddingHeadMixin, DynamicsHeadMixin


class BaseAgent(EmbeddingHeadMixin, DynamicsHeadMixin, nn.Module):
    def __init__(
        self,
        mdp_space,
        state_normalizer=None,
        embedding_head=None,
        dynamics_head=None,
        **unused,
    ):
        super().__init__()
        self.mdp_space = mdp_space
        self.max_action = mdp_space.max_action
        self.state_normalizer = state_normalizer
        self.embedding_head = embedding_head
        self.dynamics_head = dynamics_head

        self.optimize_iterations = 0

    def select_action(self, state, deterministic=None, state_embedding=None, **kwargs):
        raise NotImplementedError

    # Performs a single optimization iteration.
    # Each agent should implement how they sample from buffer and optimize any associated heads.
    def optimize(self, batch_it, batch_size, buffer, **kwargs):
        raise NotImplementedError

    # def reset_parameters(self):
    #     raise NotImplementedError

    def optimize_for(self, *args, iterations=1, batch_size=128, **kwargs):
        all_opt_info = collections.defaultdict(list)
        for batch_it in range(iterations):
            self.optimize_iterations += 1
            opt_info = self.optimize(batch_it, batch_size, *args, **kwargs)
            opt_info['iteration'] = self.optimize_iterations
            for k, v in opt_info.items():
                all_opt_info[k].append(v)
        return all_opt_info

    def state_dict(self, *args, **kwargs):
        state_dict = {}
        state_dict['optimize_iterations'] = self.optimize_iterations

        for k, v in self._modules.items():
            if isinstance(v, nn.Module) or isinstance(v, torch.optim.Optimizer):
                state_dict[k] = v.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimize_iterations = state_dict.pop('optimize_iterations')
        for k in list(state_dict.keys()):  # need to convert to list b/c modifying state_dict
            p = getattr(self, k, None)
            if isinstance(p, nn.Module) or isinstance(p, torch.optim.Optimizer):
                p.load_state_dict(state_dict.pop(k))

    # store statistics that may be unrelated to optimization
    def get_stats(self):
        return {}

    def reset_stats(self):
        pass
