from dsb.dependencies import *
import tempfile
import shutil

from .ring_storage import RingStorage


class ZarrCompactRingStorage(RingStorage):
    # assumes transitions from the same trajectory are added in order, all at once
    def __init__(
        self,
        *args,
        same_goal_in_transition=True,
        episodic_add=False,  # if True, wait until have entire episode before adding to buffer. this is useful for chunk_size > 1
        tmp_dir='',
        spool_size=int(2e9),
        # spool_size=0,
        chunk_size=1,  # chunks by batch only
        # NOTE, chunk size should be tuned according to data type and shape.
        # FB uses 128 KB w/ zstd, see https://engineering.fb.com/2018/12/19/core-data/zstandard/
        # zarr doesn't support size based chunking currently, https://github.com/zarr-developers/zarr-python/issues/270
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.same_goal_in_transition = (
            same_goal_in_transition  # NOTE: set goal to be the same within a transition, planning
        )
        self.episodic_add = episodic_add
        self.tmp_dir = tmp_dir
        self.spool_size = spool_size  # in bytes, int(1e9) bytes == 1 GB
        self.chunk_size = chunk_size

        if self.same_goal_in_transition:
            # these are the keys of data that correspond to the desired goal
            self._desired_keys = set([k for k in self.obs_space.spaces.keys() if 'desired' in k])

        self.spooled = {}
        self.filebacked = {}
        self.episodic_cache = []

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

        self._allocate_state()

    # with python3.6, there's leaked semaphore issue, see https://github.com/zarr-developers/numcodecs/issues/230
    # tried to fix this below but it doesn't solve this
    # def __del__(self):
    #     try:
    #         # numcodecs.blosc.mutex = None
    #         self.filebacked = None
    #         self.store.close()
    #         self.store = None
    #     except Exception as e:
    #         raise e
    #         pass

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['s']
        del state['filebacked']
        state.pop('store', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.s = {}
        self.filebacked = {}
        self.load_spooled()
        self.load_filebacked()

    def _allocate_state(self):
        assert self.tmp_dir is not None
        statvfs = os.statvfs(self.tmp_dir)
        nbytes_avail = statvfs.f_bavail * statvfs.f_bsize
        nbytes_used = 0

        self.filebacked_params = {}
        self.s = {}  # convenience for adding to storage

        self.terminal_mask = self.ArrayCls((self.max_size,), dtype=np.bool)

        for k, v in self.obs_space.spaces.items():
            # we compute the required storage size for k
            # if <= self.spool_size (in bytes), then we leave the array on memory
            # otherwise, fileback the array
            nbytes = int(np.prod(v.shape) * np.dtype(v.dtype).itemsize * (self.max_size + 1))
            storage_shape = (self.max_size + 1,) + v.shape

            if nbytes <= self.spool_size:
                self.spooled[k] = self.ArrayCls(storage_shape, dtype=v.dtype)
                self.spooled[k + '_terminal'] = self.ArrayCls(storage_shape, dtype=v.dtype)

                print(
                    f"spooled '{k}': nbytes={nbytes}, ngigs={nbytes / int(1e9)}, storage_shape={storage_shape}"
                )
            else:
                storage_type = v.dtype
                self.filebacked_params[k] = (storage_shape, storage_type)

                print(
                    f"filebacked '{k}': nbytes={nbytes}, ngigs={nbytes / int(1e9)}, storage_shape={storage_shape}"
                )
                nbytes_used += nbytes

        # if nbytes_used > nbytes_avail:
        #     raise Exception(f'not enough disk space, require {nbytes_used} bytes but only {nbytes_avail} available')

        self.sub_tmp_dir = tempfile.mkdtemp(dir=self.tmp_dir)
        print(f'zarr dir={self.sub_tmp_dir}')

        self.load_spooled()
        self.load_filebacked(create=True)

    def free(self):
        if self.use_shmem:
            for k, v in self.spooled.items():
                v.free()

            self.action.free()
            self.reward.free()
            self.done.free()
            # for k, v in self.state.items(): v.free()
            # for k, v in self.next_state.items(): v.free()

            # delete filebacked
            shutil.rmtree(self.sub_tmp_dir)

    def load_spooled(self):
        for k, v in self.spooled.items():
            self.s[k] = v

    @property
    def has_filebacked(self):
        return len(self.filebacked_params) > 0

    def load_filebacked(self, create=False):
        if not self.has_filebacked:
            self.store = None
            return

        import zarr

        # self.store = zarr.MemoryStore()
        # self.store = zarr.NestedDirectoryStore(self.sub_tmp_dir)
        self.store = zarr.LMDBStore(
            self.sub_tmp_dir,
            max_spare_txns=8,  # "Should match the process’s maximum expected concurrent transactions",
            # also see https://github.com/zarr-developers/zarr-python/blob/847e78a26270dbb2114771d820977eb5c2d3c5d4/zarr/storage.py#L1860
            max_readers=8,  # NOTE: should be greater than num workers if using dataloader wrapper
            lock=False,  # NOTE: not bothering locking since only main process writes, and this is separate from sampling.
            # also avoids this error: lmdb.ReadersFullError: mdb_txn_begin: MDB_READERS_FULL: Environment maxreaders limit reached
            map_size=int(500e9),  # set max db size to 500 GB
            # map_async=True, writemap=True,
            map_async=False,
            writemap=True,  # intead of map_async=True, we explicitly call store.flush(),
            # which should perform an async write to disk if necessary. We choose these settings since
            # we don't need database durability, this temp db is deleted after training
            sync=False,
            metasync=False,
            readahead=False,  # since we're dealing with random reads of individual transitions
            readonly=False if create else True,
        )  # see https://lmdb.readthedocs.io/en/release/#lmdb.Environment

        # only main process should be writing
        self.filebacked = zarr.open(store=self.store, mode='w' if create else 'r')

        if create:
            import numcodecs
            from numcodecs.blosc import NOSHUFFLE, SHUFFLE, BITSHUFFLE

            # see https://zarr.readthedocs.io/en/stable/tutorial.html#configuring-blosc
            numcodecs.blosc.use_threads = False

            for k, (storage_shape, storage_type) in self.filebacked_params.items():
                if self.chunk_size is not None:
                    chunks = (self.chunk_size,) + storage_shape[1:]  # only chunk along batch index
                else:
                    raise ValueError
                    # infer using overall shape (may split data like images into patches)
                    # chunks = zarr.util.guess_chunks(storage_shape, np.dtype(storage_type).itemsize)

                # compressor = None
                # compressor = zarr.storage.default_compressor
                # compressor = numcodecs.Blosc(cname='lz4', clevel=5, shuffle=SHUFFLE) # SHUFFLE=1, https://numcodecs.readthedocs.io/en/stable/blosc.html#numcodecs.blosc.Blosc
                # compressor = numcodecs.Blosc(cname='zstd', clevel=5, shuffle=SHUFFLE)
                compressor = numcodecs.Zstd(level=1)
                # would be cool if this supported https://github.com/facebook/zstd#the-case-for-small-data-compression

                self.filebacked.zeros(
                    name=k,
                    shape=storage_shape,
                    dtype=storage_type,
                    chunks=chunks,
                    compressor=compressor,
                )
                self.filebacked.zeros(
                    name=k + '_terminal',
                    shape=storage_shape,
                    dtype=storage_type,
                    chunks=(1,) + storage_shape[1:],
                    compressor=compressor,
                )

        for k, v in self.filebacked.items():
            self.s[k] = v

    def add_directly(self, *args, **kwargs):
        if self.episodic_add:
            return self._add_directly_episodic(*args, **kwargs)
        else:
            return self._add_directly_transition(*args, **kwargs)

    def _add_directly_episodic(self, state, action, next_state, reward, done, last_step=False):
        ptr = self.ptr

        if ptr == 0 and state['_time_step'] != 0:
            # handle special case where ptr == max_size - 1 and the trajectory has
            # not ended, so the buffer wraps to the start with ptr == 0.
            # then we need to replace s[max_size]. this is not necessary if
            # the trajectory ended w/ ptr == max_size - 1, so the current ptr
            # is the start of a new trajectory
            for k in self.obs_space.spaces.keys():
                self.s[k][self.max_size] = state[k]

        self.terminal_mask[ptr] = last_step
        if last_step:
            for k in self.obs_space.spaces.keys():
                self.s[k + '_terminal'][ptr] = next_state[k]

            cache_ptrs = [ptr]
            cache_s = {k: [v] for k, v in state.items()}

            for _ptr, _state in self.episodic_cache:
                cache_ptrs.append(_ptr)
                for k, v in _state.items():
                    cache_s[k].append(v)

            cache_ptrs = np.stack(cache_ptrs)
            cache_s = {k: np.stack(v) for k, v in cache_s.items()}

            for k in self.obs_space.spaces.keys():
                if k in self.filebacked.keys():
                    self.s[k].oindex[cache_ptrs] = cache_s[k].copy()
                else:
                    self.s[k][cache_ptrs] = cache_s[k]

            self.episodic_cache.clear()
            if hasattr(self.store, 'flush'):
                self.store.flush()  # this should be async w/ LMDBStore
        else:
            self.episodic_cache.append((ptr, copy.deepcopy(state)))

        self.action[ptr] = action
        self.reward[ptr] = reward
        self.done[ptr] = done

        self.size = min(self.size + 1, self.max_size)
        self.ptr = (self.ptr + 1) % self.max_size
        return ptr

    def _add_directly_transition(self, state, action, next_state, reward, done, last_step=False):
        ptr = self.ptr

        if ptr == 0 and state['_time_step'] != 0:
            # handle special case where ptr == max_size - 1 and the trajectory has
            # not ended, so the buffer wraps to the start with ptr == 0.
            # then we need to replace s[max_size]. this is not necessary if
            # the trajectory ended w/ ptr == max_size - 1, so the current ptr
            # is the start of a new trajectory
            for k in self.obs_space.spaces.keys():
                self.s[k][self.max_size] = state[k]

        for k in self.obs_space.spaces.keys():
            # this should overwrite next_state of the previous transition,
            # which should be same as state of the current transition to add,
            # excluding the desired goal, which is why we replace
            self.s[k][ptr] = state[k]

        self.terminal_mask[ptr] = last_step
        if last_step:  # we store the terminal observations separately
            for k in self.obs_space.spaces.keys():
                self.s[k + '_terminal'][ptr] = next_state[k]
        else:
            for k in self.obs_space.spaces.keys():
                self.s[k][ptr + 1] = next_state[k]

        self.action[ptr] = action
        self.reward[ptr] = reward
        self.done[ptr] = done

        self.size = min(self.size + 1, self.max_size)
        self.ptr = (self.ptr + 1) % self.max_size
        return ptr

    def _get_state(self, ptrs, k):
        if k in self.filebacked.keys():
            if len(ptrs.shape) == 0:
                # prevent IndexError: integer arrays in an orthogonal selection must be 1-dimensional only
                return self.filebacked[k][ptrs.tolist()]
            else:
                return self.filebacked[k].get_orthogonal_selection(ptrs)
        else:
            return self.spooled[k][ptrs]

    def get_next_state(self, ptrs, k_=None):
        # need to check if ptr is terminal and deal separately
        ptrs = np.array(ptrs)
        # terminal_mask = np.isin(ptrs, list(self.filebacked['terminal'].keys())) # converting to list since np.isin doesn't work with sets
        terminal_mask = self.terminal_mask[ptrs]
        terminal_ptrs = ptrs[terminal_mask]

        if k_ is not None:
            if k_.startswith('c_'):
                assert self.with_context
                k = k_[2:]
                s = self.get_context(
                    [k],
                    ptrs,
                    which='state' if self.same_goal_in_transition else 'next_state',
                )[k]
            # elif k_.startswith('fs_'): # TODO: check if same_goal_in_transition
            #     assert self.framestack_timesteps is not None
            #     k = k_[3:]
            #     s = self.get_framestack([k], ptrs, self.framestack_timesteps, which='next_state')[k]
            else:
                if self.same_goal_in_transition and k_ in self._desired_keys:
                    s = self._get_state(ptrs, k_)
                else:
                    s = self._get_state(ptrs + 1, k_)

                    if len(terminal_ptrs) > 0:
                        s[terminal_mask] = self._get_state(terminal_ptrs, k_ + '_terminal')
        else:
            s = {}
            _keys = self.obs_space.spaces.keys()
            if self.same_goal_in_transition:
                _keys = filter(lambda x: x not in self._desired_keys, _keys)
            for k in _keys:
                v = self._get_state(ptrs + 1, k)
                if len(terminal_ptrs) > 0:
                    v[terminal_mask] = self._get_state(terminal_ptrs, k + '_terminal')
                s[k] = v

            if self.same_goal_in_transition:
                for k in self._desired_keys:
                    v = self._get_state(ptrs, k)
                    s[k] = v

            if self.with_context:
                for k in self.context_type.keys():
                    v = self.get_context(
                        [k],
                        ptrs,
                        which='state' if self.same_goal_in_transition else 'next_state',
                    )[k]
                    s[f'c_{k}'] = v
        return s
