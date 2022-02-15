from dsb.dependencies import *
from dsb.buffers.utils import get_timestep_of, sample_pos_pair_indices
from dsb.utils import torchify

from ..sample_buffer_wrapper import SampleBufferWrapper


class OnlineGCBCBufferWrapper(SampleBufferWrapper):
    # given a padded sequence batch of full trajectories
    # or a window of trajectories (B, T, ...)
    # extracts the final state of the window
    # as the goal for all transitions from the window
    def __init__(
        self,
        buffer,
        **kwargs,
    ):
        super().__init__(buffer, **kwargs)

    def get_batch(self, batch_it, ptrs):
        ptrs, batch = self.buffer.get_batch(batch_it, ptrs)
        state, action, next_state, reward, done = batch

        _length = state.pop('_length')

        B = len(_length)

        # TODO: should we be getting next_state instead as goal?
        goal = state['achieved_goal'][torch.arange(B), torch.tensor(_length) - 1, ...]

        # it's ok to fill padded timesteps since they will just get masked out
        # broadcast so replacing all timesteps
        goal = goal.unsqueeze(1)
        state['desired_goal'][:, :, ...] = goal
        next_state['desired_goal'][:, :, ...] = goal

        # TODO: add relabel mask? just all timesteps

        state['_length'] = _length

        return ptrs, batch
