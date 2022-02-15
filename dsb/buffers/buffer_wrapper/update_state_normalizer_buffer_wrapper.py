from dsb.dependencies import *
from dsb.utils import torchify

from .buffer_wrapper import BufferWrapper


class UpdateStateNormalizerBufferWrapper(BufferWrapper):
    """update immediately w/ a single transition"""

    def __init__(self, buffer, update_with_terminal=True, state_normalizer=None, **kwargs):
        super().__init__(buffer, **kwargs)
        self.update_with_terminal = update_with_terminal  # faster without
        assert state_normalizer is not None
        self._update_fn = state_normalizer.update_stats

    def add(self, *transition, last_step=False, env_idx=None):
        ptr = self.buffer.add(*transition, last_step=last_step, env_idx=env_idx)

        state, action, next_state, reward, done = transition

        self._update_fn(torchify(state))
        if last_step and self.update_with_terminal:
            self._update_fn(torchify(next_state))

        return ptr


class TrajectoryUpdateStateNormalizerBufferWrapper(BufferWrapper):
    """wait until episode trajectory is completely added before updating"""

    def __init__(self, buffer, update_with_terminal=True, state_normalizer=None, **kwargs):
        super().__init__(buffer, **kwargs)
        self.update_with_terminal = update_with_terminal  # faster without
        assert state_normalizer is not None
        self._update_fn = state_normalizer.update_stats
        self._requires_update = state_normalizer.requires_update

        self.episode_transition_ptrs = collections.defaultdict(list)

    def add(self, *transition, last_step=False, env_idx=None):
        ptr = self.buffer.add(*transition, last_step=last_step, env_idx=env_idx)

        self.episode_transition_ptrs[env_idx].append(ptr)

        if last_step:
            inds = np.array(self.episode_transition_ptrs[env_idx])

            # check which keys actually require an update
            s = {}
            s_terminal = {}
            for k in self.buffer.obs_space.spaces.keys():
                if self._requires_update.get(k, False):
                    s[k] = self.buffer.get_state(inds, k_=k).copy()

                    if self.update_with_terminal:
                        s_terminal[k] = self.buffer.get_next_state(ptr, k_=k).copy()

            if self.update_with_terminal:
                s_all = {}
                for k in s.keys():
                    a = s[k].copy()
                    b = s_terminal[k].copy()
                    s_all[k] = np.concatenate([a, [b]])
            else:
                s_all = s

            self._update_fn(torchify(s_all))
            self.episode_transition_ptrs[env_idx].clear()

        return ptr
