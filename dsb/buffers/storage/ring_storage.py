from dsb.dependencies import *
from ..shmem_numpy_array import ShmemNumpyArray
from .base_storage import BaseStorage

from dsb.buffers.utils import get_timestep_of

# max_size is based on the number of (s, a, r, s') transitions
class RingStorage(BaseStorage):
    def __init__(
        self,
        obs_space,
        action_space,
        max_size,
        use_shmem=False,
        batch_dtype=None,
        framestack_keys=['achieved_goal'],
        framestack_timesteps=None,
        with_context=False,
        # context_type=dict(achieved_goal='episode_start', desired_goal='prev_goal'),
        context_type=dict(achieved_goal='prev_goal', desired_goal='prev_goal'),
        **unused,
    ):
        super().__init__(obs_space, action_space, max_size, batch_dtype=batch_dtype)
        self.use_shmem = use_shmem  # set to True if using dataloader wrapper

        self.size = 0
        self.ptr = 0

        self.framestack_keys = set(framestack_keys)
        if framestack_timesteps is not None:
            self.framestack_timesteps = np.array(framestack_timesteps)
        else:
            self.framestack_timesteps = None
        self.prev_goal_ptrs = None  # saves ptr corresponding to timestep of prev subgoal
        self.with_context = with_context
        self.context_type = context_type

        # populated by episodic_replay_buffer
        self.episode_ptrs = {}
        self.start_ptrs = None
        self.invalid_mask = None

    @property
    def ArrayCls(self):
        return ShmemNumpyArray if self.use_shmem else np.zeros

    def allocate(self):
        assert isinstance(self.obs_space, gym.spaces.Dict)
        if isinstance(self.action_space, gym.spaces.Discrete):
            action_dim = 1
            action_dtype = np.int32
        elif isinstance(self.action_space, gym.spaces.Box):
            action_dim = self.action_space.shape[0]
            action_dtype = np.float32
        else:
            raise ValueError

        ArrayCls = self.ArrayCls

        self.action = ArrayCls((self.max_size, action_dim), dtype=action_dtype)
        self.reward = ArrayCls((self.max_size, 1), dtype=np.float32)
        self.done = ArrayCls((self.max_size, 1), dtype=np.bool)

        # NOTE: np.zeros will NOT preallocate the array in memory (np.ones does though)
        # this can be problematic if your machine's memory is insufficient to hold
        # the replay buffer data, and the program may be killed due to running out of memory.
        # to preallocate, do 0 * np.ones()
        self.state = {
            k: ArrayCls((self.max_size,) + v.shape, dtype=v.dtype)
            for k, v in self.obs_space.spaces.items()
        }
        self.next_state = {
            k: ArrayCls((self.max_size,) + v.shape, dtype=v.dtype)
            for k, v in self.obs_space.spaces.items()
        }

    def free(self):
        if self.use_shmem:
            self.action.free()
            self.reward.free()
            self.done.free()
            for k, v in self.state.items():
                v.free()
            for k, v in self.next_state.items():
                v.free()

    def add_directly(self, state, action, next_state, reward, done, last_step=False):
        ptr = self.ptr

        for k in self.obs_space.spaces.keys():
            self.state[k][ptr] = state[k]
            self.next_state[k][ptr] = next_state[k]

        self.action[ptr] = action
        self.reward[ptr] = reward
        self.done[ptr] = done

        self.size = min(self.size + 1, self.max_size)
        self.ptr = (self.ptr + 1) % self.max_size
        return ptr

    def get_context(self, keys, ptrs, which='state'):
        s = {}
        for k_ in keys:
            c = self.context_type[k_]
            if c == 'prev_goal':
                selected_ptrs = self.prev_goal_ptrs[ptrs]
            elif c == 'episode_start':
                selected_ptrs = self.start_ptrs[ptrs]
            else:
                raise ValueError

            if which == 'state':
                s[k_] = self.get_state(selected_ptrs, k_=k_)
            elif which == 'next_state':
                raise NotImplementedError  # untested, pretty sure this isn't right
                s[k_] = self.get_next_state(selected_ptrs, k_=k_)
            else:
                raise NotImplementedError
        return s

    def get_framestack(self, keys, ptrs, fs_t, which='state'):
        all_fs_ptrs = []
        for ptr in ptrs:  # need to iterate over the batch here
            start_ptr = self.start_ptrs[ptr]
            episode_ptrs = np.array(self.episode_ptrs[start_ptr][:])

            t = get_timestep_of([ptr], episode_ptrs)[0]
            fs_indices = (t + fs_t).clip(0, len(episode_ptrs) - 1)
            fs_ptrs = episode_ptrs[fs_indices]

            all_fs_ptrs.append(fs_ptrs)

        s = {}
        B = len(ptrs)
        T = len(fs_t)  # could also just use -1
        for k_ in keys:
            if which == 'state':
                v = self.get_state(all_fs_ptrs, k_=k_)
            elif which == 'next_state':
                # TODO: increment fs_t by 1 and just use get_state
                raise NotImplementedError
                v = self.get_next_state(all_fs_ptrs, k_=k_)
            else:
                raise ValueError

            # this reshaping should associate the
            # right timesteps w/ the corresponding batch elements
            # since all_fs_ptrs is ordered by batch element
            s[k_] = v.reshape((B, T) + v.shape[1:])  # (B, T, ....)
        return s

    def _get_state(self, ptrs, k):
        return self.state[k][ptrs]

    def _get_next_state(self, ptrs, k):
        return self.next_state[k][ptrs]

    def get_state(self, ptrs, k_=None):
        ptrs = np.array(ptrs)
        if k_ is not None:
            if k_.startswith('c_'):
                assert self.with_context
                k = k_[2:]
                s = self.get_context([k], ptrs, which='state')[k]
            elif k_.startswith('fs_'):
                assert self.framestack_timesteps is not None
                k = k_[3:]
                s = self.get_framestack([k], ptrs, self.framestack_timesteps, which='state')[k]
            else:
                s = self._get_state(ptrs, k_)
        else:
            s = {}
            for k, v in self.obs_space.spaces.items():
                s[k] = self._get_state(ptrs, k)

            if self.with_context:
                c_s = self.get_context(self.context_type.keys(), ptrs, which='state')
                for k, v in c_s.items():
                    s[f'c_{k}'] = v

            if self.framestack_timesteps is not None:
                fs_s = self.get_framestack(
                    self.framestack_keys, ptrs, self.framestack_timesteps, which='state'
                )
                for k, v in fs_s.items():
                    s[f'fs_{k}'] = v
        return s

    def get_next_state(self, ptrs, k_=None):
        ptrs = np.array(ptrs)
        if k_ is not None:
            if k_.startswith('c_'):
                assert self.with_context
                k = k_[2:]
                s = self.get_context([k], ptrs, which='next_state')[k]
            # elif k_.startswith('fs_'):
            #     assert self.framestack_timesteps is not None
            #     k = k_[3:]
            #     s = self.get_framestack([k], ptrs, self.framestack_timesteps, which='next_state')[k]
            else:
                s = self._get_next_state(ptrs, k_)
        else:
            s = {}
            for k, v in self.obs_space.spaces.items():
                s[k] = self._get_next_state(ptrs, k)

            if self.with_context:
                c_s = self.get_context(self.context_type.keys(), ptrs, which='next_state')
                for k, v in c_s.items():
                    s[f'c_{k}'] = v

            # if self.framestack_timesteps is not None:
            #     fs_s = self.get_framestack(self.framestack_keys, ptrs, self.framestack_timesteps, which='next_state')
            #     for k, v in fs_s.items():
            #         s[f'fs_{k}'] = v
        return s

    def get_batch(self, ptrs):  # this must be process safe
        state = self.get_state(ptrs)  # this should include framestack
        next_state = self.get_next_state(ptrs)

        if self.batch_dtype is not None:  # force a copy
            state = {k: v.astype(self.batch_dtype, copy=True) for k, v in state.items()}
            next_state = {k: v.astype(self.batch_dtype, copy=True) for k, v in next_state.items()}
            action = self.action[ptrs].astype(self.batch_dtype, copy=True)
            reward = self.reward[ptrs].astype(self.batch_dtype, copy=True)
            done = self.done[ptrs].astype(self.batch_dtype, copy=True)
        else:
            state = {k: v.copy() for k, v in state.items()}
            next_state = {k: v.copy() for k, v in next_state.items()}
            action = self.action[ptrs].copy()
            reward = self.reward[ptrs].copy()
            done = self.done[ptrs].copy()

        batch = (
            state,
            action,
            next_state,
            reward,
            done,
        )
        return ptrs, batch
