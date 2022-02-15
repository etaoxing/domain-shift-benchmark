from dsb.dependencies import *

from .buffer_wrapper import BufferWrapper


class TrajectoryAddBufferWrapper(BufferWrapper):
    def __init__(self, buffer, **kwargs):
        super().__init__(buffer, **kwargs)

        # temporary buffer for storing transitions of trajectories, key is env_idx
        self.episode_transitions = collections.defaultdict(list)

    def add(self, *transition, last_step=False, env_idx=None):
        self.episode_transitions[env_idx].append(transition)
        if last_step:
            self.store_episode(env_idx)
            self.episode_transitions[env_idx].clear()

        return None

    def store_episode(self, env_idx):
        transition_idx = 0
        num_transitions = len(self.episode_transitions[env_idx])

        while transition_idx < num_transitions:
            transition = self.episode_transitions[env_idx][transition_idx]
            last_step = transition_idx == (num_transitions - 1)
            ptr = self.buffer.add(*transition, last_step=last_step)

            transition_idx += 1
