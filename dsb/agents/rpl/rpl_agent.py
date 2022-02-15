from dsb.dependencies import *
import dsb.builder as builder
from dsb.utils import torchify, untorchify

from ..base_agent import BaseAgent


class TempBuffer:
    def __init__(self, ptrs, batch):
        self.ptrs = ptrs
        self.batch = batch
        self.batch_dtype = None

    def sample(self, batch_it, batch_size):
        ptrs = self.ptrs
        batch = self.batch

        self.ptrs = None
        self.batch = None

        return ptrs, batch


# create a tied copy of DictNormalizer, so underlying objects are the same
def create_tied_state_normalizer(dict_state_normalizer, ignore_keys=[]):
    from dsb.normalizers import DictNormalizer

    assert isinstance(dict_state_normalizer, DictNormalizer)
    ignore_keys_set = set(ignore_keys)

    normalizers = {
        k: v for k, v in dict_state_normalizer.normalizers.items() if k not in ignore_keys_set
    }
    normalizers_params = {
        k: v
        for k, v in dict_state_normalizer.normalizers_params.items()
        if k not in ignore_keys_set
    }
    return DictNormalizer(normalizers, normalizers_params)


class RPLAgent(BaseAgent):
    # see figure 3 of RPL paper
    # optimization is decoupled

    def __init__(
        self,
        mdp_space,
        agent_params={},
        share_embedding_head=False,
        high_optimize_interval=1,
        low_optimize_interval=1,
        **kwargs,
    ):
        super().__init__(mdp_space, **kwargs)
        self.share_embedding_head = share_embedding_head

        # build high level agent
        high_mdp_space = copy.deepcopy(mdp_space)
        if self.embedding_head:
            high_mdp_space.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(self.embedding_head.embedding_dim,)
            )
        else:
            high_mdp_space.action_space = copy.deepcopy(mdp_space.obs_space['desired_goal'])
            high_mdp_space.action_space = gym.spaces.Box(
                low=-1, high=1, shape=mdp_space.obs_space['desired_goal'].shape
            )
        self.high_agent = builder.build_agent(
            agent_params,
            high_mdp_space,
            embedding_head=self.embedding_head,
            state_normalizer=self.state_normalizer,
        )

        # build low level agent
        # duplicates state_normalizer to ignore high_action / low_desired_goal
        if self.embedding_head:
            if self.share_embedding_head:
                raise NotImplementedError
            else:
                low_embedding_head = builder.clone_module(
                    self.embedding_head, reset_parameters=True
                )
        else:
            self.share_embedding_head = False
            low_embedding_head = None
        low_state_normalizer = create_tied_state_normalizer(
            self.state_normalizer, ignore_keys=['desired_goal']
        )
        self.low_agent = builder.build_agent(
            agent_params,
            mdp_space,
            embedding_head=low_embedding_head,
            state_normalizer=low_state_normalizer,
        )

        # delete unused references
        self.embedding_head = None
        self.state_normalizer = None

    def select_action(self, state, high_action_mask=None, state_embedding=None, **kwargs):
        with torch.no_grad():
            # if high_action is provided, then we allow masking so that high_action
            # may be kept the same by a policy if desired
            high_action = state.pop('high_action', None)
            high_state, new_high_action = self.high_agent.select_action(
                state, state_embedding=state_embedding, **kwargs
            )

            high_goal = state.pop('desired_goal')

            if self.high_agent.embedding_head:
                if high_action.shape != new_high_action.shape:
                    _ha = torchify(dict(desired_goal=high_action))
                    high_action = self.high_agent.compute_embedding(_ha)['desired_goal']
                    new_high_action = torchify(new_high_action)

                if high_action_mask is not None:
                    high_action[high_action_mask, ...] = new_high_action[high_action_mask, ...]
                else:
                    high_action = new_high_action

                low_state_embedding = torchify(dict(desired_goal=high_action))
            else:
                if high_action_mask is not None:
                    high_action[high_action_mask, ...] = new_high_action[high_action_mask, ...]
                else:
                    high_action = new_high_action

                # swap in high_action as goal for low_agent and get low_action
                state['desired_goal'] = high_action

                low_state_embedding = None

            low_state, low_action = self.low_agent.select_action(
                state, state_embedding=low_state_embedding, **kwargs
            )

            # undo changes to state
            state['desired_goal'] = high_goal
            state['high_action'] = untorchify(high_action)
            return state, low_action

    def optimize(self, batch_it, batch_size, buffer):
        opt_info = {}

        low_buffer, high_buffer = self.get_low_and_high_buffers(batch_it, batch_size, buffer)

        self.high_agent.optimize_iterations += 1  # manually inc b/c this is done in optimize_for
        high_opt_info = self.high_agent.optimize(batch_it, batch_size, high_buffer)
        for k, v in high_opt_info.items():
            opt_info[f'high_{k}'] = v

        self.low_agent.optimize_iterations += 1  # manually inc b/c this is done in optimize_for
        low_opt_info = self.low_agent.optimize(batch_it, batch_size, low_buffer)
        for k, v in low_opt_info.items():
            opt_info[f'low_{k}'] = v

        # TODO: equations 2 and 3, separate sampling of buffer into demos and online data

        # section 5.2, "sufficient to fine-tune the low-level policy, although we could also
        # fine-tune both levels, at the cost of more non-stationarity", for the RL finetuning stage

        return opt_info

    def get_low_and_high_buffers(self, batch_it, batch_size, buffer):
        ptrs, batch = buffer.sample(batch_it, batch_size)
        batch_dtype = None if buffer.batch_dtype is not None else torch.float32
        batch = tuple(torchify(x, dtype=batch_dtype) for x in batch)

        state, action, next_state, reward, done = batch

        low_goal = state.pop('low_desired_goal')
        high_goal = state.pop('desired_goal')
        high_action = state.pop('high_action')

        if self.high_agent.embedding_head:
            x = dict(achieved_goal=high_action)
            # need to embed high_action
            high_action = self.high_agent.compute_embedding(x)['achieved_goal']
        else:
            if self.high_agent.state_normalizer:
                high_action = self.high_agent.state_normalizer(high_action, which='achieved_goal')

        next_low_goal = next_state.pop('low_desired_goal')
        next_high_goal = next_state.pop('desired_goal')

        # create low and high batch
        low_state = {k: v for k, v in state.items()}
        low_state['desired_goal'] = low_goal
        low_next_state = {k: v for k, v in low_state.items()}
        low_next_state['desired_goal'] = next_low_goal

        low_batch = (low_state, action, low_next_state, reward, done)

        high_state = {k: v for k, v in state.items()}
        high_state['desired_goal'] = high_goal
        high_next_state = {k: v for k, v in next_state.items()}
        high_next_state['desired_goal'] = next_high_goal

        high_batch = (high_state, high_action, high_next_state, reward, done)

        # use temp buffer that just returns the given batch
        low_buffer = TempBuffer(ptrs, low_batch)
        high_buffer = TempBuffer(ptrs, high_batch)

        return low_buffer, high_buffer
