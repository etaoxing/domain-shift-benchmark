from dsb.dependencies import *

from .replay_buffer import ReplayBuffer


class ShareableList(object):
    def __init__(self, ArrayCls, dtype, max_size, lock=None):
        if lock is not None:
            self.array = ArrayCls(max_size, dtype=dtype, lock=lock)
        else:
            self.array = ArrayCls(max_size, dtype=dtype)
        self.length = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.array[: self.length][idx]

    def append(self, value):
        self.array[self.length] = value
        self.length += 1

    def free(self):
        self.array.free()
        self.array = None


class EpisodicReplayBuffer(ReplayBuffer):
    # Tracks which ptrs belong to each episode trajectory
    # assumes that transitions for same episode are added sequentially
    # use in combination w/ TrajectoryAddBufferWrapper

    def __init__(self, *args, max_episode_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        # self.buffer should be a storage object

        # TODO: to save on some memory, offset start_ptr by +1
        self.buffer.start_ptr = self.buffer.ArrayCls(
            (self.buffer.max_size,), dtype=np.int32
        )  # track which episode each transition belongs to
        self.buffer.start_ptr[:] = -1

        # moved this into storage
        # self.episode_ptrs = {} # lists should be temporally ordered, key is start_ptr
        # self.ShareableList = functools.partial(ShareableList, self.buffer.ArrayCls, np.int32, max_episode_steps)
        # causes os error, too many files open
        # if self.use_shmem:
        #     import multiprocessing
        #     self.episode_ptrs_lock = multiprocessing.Lock()
        # else:
        #     self.episode_ptrs_lock = None

        # from multiprocessing import Manager
        # self.shareable_list_manager = Manager() # can't use since this gives threading.Lock()

        self.buffer.invalid_mask = self.buffer.ArrayCls(
            (self.buffer.max_size,), dtype=np.bool
        )  # could also just check where start ptr is null but we need this mask when sampling

        self.cur_start_ptr = None
        self.looped_once = False

        self.save_prev_goal_ptr = self.buffer.with_context
        if self.save_prev_goal_ptr:
            self.prev_ptr = None
            self.prev_goal_ptr = None
            self.buffer.prev_goal_ptrs = self.buffer.ArrayCls(
                (self.buffer.max_size,), dtype=np.int32
            )  # track the ptr of the previous subgoal
            self.buffer.prev_goal_ptrs[:] = -1

    def free(self):
        if self.use_shmem:
            self.buffer.start_ptr.free()
            # for k, v in self.buffer.episode_ptrs.items(): v.free()
            self.buffer.invalid_mask.free()
            self.buffer.free()

    def delete_transitions_from_episode(self, ptr):
        if self.buffer.invalid_mask[ptr]:
            return  # if already invalid, then don't need to do anything

        # this is the previous trajectory that we want to overwrite
        start_ptr = self.buffer.start_ptr[ptr]
        _ep_ptrs = self.buffer.episode_ptrs.pop(start_ptr)
        ep_ptrs = _ep_ptrs[:]

        self.buffer.start_ptr[ep_ptrs] = -1
        self.buffer.invalid_mask[ep_ptrs] = True

        if self.save_prev_goal_ptr:
            self.buffer.prev_goal_ptrs[ep_ptrs] = -1

        # if self.use_shmem:
        #     _ep_ptrs.free()

    def add(self, state, action, next_state, reward, done, last_step=False, env_idx=None):
        ptr = self.buffer.add_directly(state, action, next_state, reward, done, last_step=last_step)

        if self.looped_once:
            self.delete_transitions_from_episode(ptr)
        elif self.at_capacity:
            self.looped_once = True
            # setting this so when we go back to the start, we begin using the invalid mask.
            # could also just check at_capacity, but this is true when ptr == max_size - 1
            # on the first loop, since we just added to the buffer

        if self.cur_start_ptr is None:
            self.cur_start_ptr = ptr

            self.prev_ptr = ptr
            self.prev_goal_ptr = ptr
            assert self.cur_start_ptr not in self.buffer.episode_ptrs.keys()
            # self.buffer.episode_ptrs[self.cur_start_ptr] = self.ShareableList(lock=self.episode_ptrs_lock)
            self.buffer.episode_ptrs[self.cur_start_ptr] = []
        assert self.buffer.start_ptr[ptr] == -1
        # these asserts check that the space is freely available to write to

        self.buffer.start_ptr[ptr] = self.cur_start_ptr
        self.buffer.episode_ptrs[self.cur_start_ptr].append(ptr)

        if self.save_prev_goal_ptr:
            assert self.buffer.prev_goal_ptrs[ptr] == -1
            if state.get('_subgoal_reached', False):  # given a new subgoal, so update context
                self.prev_goal_ptr = self.prev_ptr
            self.prev_goal_ptrs[ptr] = self.prev_goal_ptr
        self.prev_ptr = ptr

        if self.looped_once:
            self.buffer.invalid_mask[ptr] = False

        if last_step:
            self.cur_start_ptr = None

            self.prev_ptr = None
            self.prev_goal_ptr = None

        return ptr

    @property
    def at_capacity(self):
        return self.buffer.size == self.buffer.max_size

    def sample_ptrs(self, batch_it, batch_size):
        if not self.at_capacity:
            return super().sample_ptrs(batch_it, batch_size)

        try:
            r = np.arange(0, self.buffer.max_size)[
                ~self.buffer.invalid_mask[:]
            ]  # NOTE: using [:] to get numpy array from ShmemNumpyArray
            ptrs = r[np.rng.integers(0, len(r), size=batch_size)]
        except ValueError as e:
            raise ValueError(f'Need more samples in buffer, {e}')

        return ptrs
