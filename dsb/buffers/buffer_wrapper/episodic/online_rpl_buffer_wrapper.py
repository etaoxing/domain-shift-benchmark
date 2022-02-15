from dsb.dependencies import *
from dsb.buffers.utils import get_timestep_of, sample_pos_pair_indices
from dsb.utils import torchify

from ..sample_buffer_wrapper import SampleBufferWrapper


class OnlineRPLBufferWrapper(SampleBufferWrapper):
    def __init__(
        self,
        buffer,
        # see appendix A of https://arxiv.org/abs/1910.11956
        window_low=30,
        window_high=260,
        high_action_same_as_low_goal=False,
        substitute_tensor=False,  # set True w/ demonstrations replay buffe
        **kwargs,
    ):
        super().__init__(buffer, **kwargs)
        self.window_low = window_low
        self.window_high = window_high
        self.high_action_same_as_low_goal = high_action_same_as_low_goal
        self.substitute_tensor = substitute_tensor

    def get_batch(self, batch_it, ptrs):
        ptrs, batch = self.buffer.get_batch(batch_it, ptrs)
        state, action, next_state, reward, done = batch

        # see Algorithm 2 and 3 of https://arxiv.org/abs/1910.11956
        # we implement relay data relabeling in an online fashion

        # and the low goal as state['low_desired_goal']
        # setting the high goal as state['desired_goal']
        # setting the high action as state['high_action']

        low_goal_all_selected_episode_ptrs = []
        high_goal_all_selected_episode_ptrs = []
        high_action_all_selected_episode_ptrs = []

        start_ptrs = self.start_ptr[ptrs]
        for s in np.unique(start_ptrs):
            # NOTE: using [:] to get numpy array from ShmemNumpyArray
            episode_ptrs = np.array(self.episode_ptrs[s][:])
            N = len(episode_ptrs)
            if N == 1:
                continue  # if t=[0] and N=1, then no future transitions

            # gets timestep of ptr in episode transitions
            y = ptrs[start_ptrs == s]
            t = get_timestep_of(y, episode_ptrs)

            # if finding pair for last transition, then it will be paired with itself
            low_goal_selected_t = sample_pos_pair_indices(t, N, 1, self.window_low)
            low_goal_selected_episode_ptr = episode_ptrs[low_goal_selected_t]
            low_goal_all_selected_episode_ptrs.append(low_goal_selected_episode_ptr)

            # if finding pair for last transition, then it will be paired with itself
            high_goal_selected_t = sample_pos_pair_indices(t, N, 1, self.window_high)
            high_goal_selected_episode_ptr = episode_ptrs[high_goal_selected_t]
            high_goal_all_selected_episode_ptrs.append(high_goal_selected_episode_ptr)

            if self.high_action_same_as_low_goal:
                pass  # dealt with later
            else:
                # if finding pair for last transition, then it will be paired with itself
                high_action_selected_t = sample_pos_pair_indices(t, N, 1, self.window_low)
                high_action_selected_episode_ptr = episode_ptrs[high_action_selected_t]
                high_action_all_selected_episode_ptrs.append(high_action_selected_episode_ptr)

        low_goal_selected_episode_ptr = np.concatenate(low_goal_all_selected_episode_ptrs)
        high_goal_selected_episode_ptr = np.concatenate(high_goal_all_selected_episode_ptrs)
        if self.high_action_same_as_low_goal:
            high_action_selected_episode_ptr = None
        else:
            high_action_selected_episode_ptr = np.concatenate(high_action_all_selected_episode_ptrs)

        batch = self.substitute_goal(
            batch,
            low_goal_selected_episode_ptr,
            high_goal_selected_episode_ptr,
            high_action_selected_episode_ptr,
        )
        return ptrs, batch

    def substitute_goal(
        self, batch, low_goal_relabel_ptrs, high_goal_relabel_ptrs, high_action_relabel_ptrs
    ):
        state, action, next_state, reward, done = batch

        # TODO: should we be getting next_state instead as goal?
        _low_goal = self.buffer.get_state(low_goal_relabel_ptrs, 'achieved_goal').copy()
        if self.substitute_tensor:
            low_goal = torchify(_low_goal, dtype=None, device='cpu')
        else:
            low_goal = _low_goal
        state['low_desired_goal'] = low_goal
        next_state['low_desired_goal'] = low_goal

        high_goal = self.buffer.get_state(high_goal_relabel_ptrs, 'achieved_goal').copy()
        if self.substitute_tensor:
            low_goal = torchify(low_goal, dtype=None, device='cpu')
        state['desired_goal'] = high_goal
        next_state['desired_goal'] = high_goal

        if self.high_action_same_as_low_goal:
            if self.substitute_tensor:
                high_action = torchify(_low_goal.copy(), dtype=None, device='cpu')
            else:
                high_action = low_goal.copy()
        else:
            high_action = self.buffer.get_state(high_action_relabel_ptrs, 'achieved_goal').copy()
            if self.substitute_tensor:
                high_action = torchify(high_action, dtype=None, device='cpu')
        state['high_action'] = high_action

        batch = (state, action, next_state, reward, done)
        return batch
