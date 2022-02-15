from dsb.dependencies import *
from torch.utils.data import DataLoader, Sampler

from .buffer_wrapper import SampleBufferWrapper
from .utils import worker_init_fn, sample_context_indices


class DemoTrajectoryBatchSampler(Sampler):
    def __init__(
        self,
        data_source,
        episode_ptrs,
        batch_num_trajectories=None,
        drop_last=False,
        window_bounds=None,
    ):
        super().__init__(data_source)
        self.episode_ptrs = episode_ptrs
        self.num_trajectories = len(self.episode_ptrs.keys())
        self.batch_num_trajectories = batch_num_trajectories
        self.drop_last = drop_last

        # this is an inclusive interval [a, b]
        if isinstance(window_bounds, int):
            self.window_bounds = (window_bounds, window_bounds)
        elif isinstance(window_bounds, list) or isinstance(window_bounds, tuple):
            assert len(window_bounds) == 2
            self.window_bounds = window_bounds
        else:
            self.window_bounds = window_bounds

        if self.batch_num_trajectories is None:
            self.batch_num_trajectories = 1

        assert self.num_trajectories == data_source.num_demos

        self.num_samples = self.num_trajectories // self.batch_num_trajectories
        if not self.drop_last and self.num_trajectories % self.batch_num_trajectories != 0:
            self.num_samples += 1

    def __iter__(self):
        l = list(self.episode_ptrs.keys())
        np.rng.shuffle(l)

        s = 0
        while s < self.num_samples:
            p = []
            for n in range(self.batch_num_trajectories):
                i = s * self.batch_num_trajectories + n
                if i < len(l):
                    ep_ptrs = copy.deepcopy(self.episode_ptrs[l[i]])

                    N = len(ep_ptrs)
                    if self.window_bounds is not None:
                        # pick a random length for the window
                        w = np.rng.integers(
                            self.window_bounds[0], self.window_bounds[1], endpoint=True
                        )

                        # pick a random starting index of window
                        t = np.rng.integers(0, N - w, endpoint=True)

                        selected_ptrs = ep_ptrs[t : t + w]

                        if len(selected_ptrs) != w:
                            raise RuntimeError
                    else:  # use full trajectory
                        selected_ptrs = ep_ptrs

                    p += selected_ptrs

            s += 1
            yield p

    def __len__(self):
        return self.num_samples


class DemonstrationsBuffer(SampleBufferWrapper):
    def __init__(
        self,
        *args,
        datasets=None,
        num_workers=0,
        #
        sample_mode='transition',
        batch_size_opt=None,
        subsample_batch_size=None,
        window_bounds=None,
        set_context_with_batch=False,
        T_context=None,  # number of context frames sampled from video
        framestack_keys=set(['achieved_goal']),
        framestack_timesteps=None,
        **unused,
    ):
        self.batch_dtype = None
        super().__init__(*args)
        self.datasets = datasets
        self.dataset = None  # currently active dataset

        self.num_workers = num_workers
        self.sample_mode = sample_mode
        self.batch_size_opt = batch_size_opt
        self.subsample_batch_size = subsample_batch_size
        self.window_bounds = window_bounds
        self.set_context_with_batch = set_context_with_batch
        self.T_context = T_context
        self.framestack_keys = framestack_keys
        if framestack_timesteps is not None:
            framestack_timesteps = np.array(framestack_timesteps)
        self.framestack_timesteps = framestack_timesteps

    def set_batch_size_opt(self, batch_size_opt):
        if batch_size_opt != self.batch_size_opt:
            self.batch_size_opt = batch_size_opt
            # Need to re-initialize dataloader if changing the batch size
            self._make_dataloader()

    def init_dataloader(self, dataset_key):
        assert self.datasets is not None
        self.dataset_key = dataset_key
        self.dataset = self.datasets[dataset_key]
        self._make_dataloader()

    def _make_dataloader(self):
        if self.framestack_timesteps is not None:
            self.dataset.set_framestack_params(self.framestack_keys, self.framestack_timesteps)

        self.start_ptr = np.ones(len(self.dataset), dtype=np.int) * -1
        self.episode_ptrs = collections.defaultdict(list)

        i = 0
        for demo in self.dataset.demos:
            s = i
            for _ in range(demo['num_timesteps']):
                self.episode_ptrs[s].append(i)
                self.start_ptr[i] = s
                i += 1

        if self.sample_mode == 'transition':  # batch is transitions from random trajectories
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size_opt,
                drop_last=True,  # NOTE: dropping last batch so all batches are same size (makes logging easier, sampling neg pairs)
                shuffle=True,
                pin_memory=torch.cuda.is_available(),
                num_workers=self.num_workers,
                worker_init_fn=worker_init_fn if self.num_workers else None,
            )
        elif self.sample_mode == 'trajectory':  # batch is transitions from same trajectory
            self.dataloader = DataLoader(
                self.dataset,
                batch_sampler=DemoTrajectoryBatchSampler(
                    self.dataset,
                    self.episode_ptrs,
                    batch_num_trajectories=self.batch_size_opt,
                    window_bounds=self.window_bounds,
                    drop_last=True,
                ),
                pin_memory=torch.cuda.is_available(),
                num_workers=self.num_workers,
                worker_init_fn=worker_init_fn if self.num_workers else None,
            )
            print('Using trajectory sampler, batch_size_opt sets the number of trajectories.')
        else:
            raise ValueError

        self.iter = iter(self.dataloader)
        self.size = len(self.iter)

        if self.size <= 0:
            raise RuntimeError('No batches, batch size probably greater than dataset length.')

        self.epoch = 0
        self.batches = {}  # cache

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['iter']
        return state

    def __setstate__(self, state):
        state['iter'] = None
        self.__dict__.update(state)

    def sample_ptrs(self, batch_it, batch_size):
        tries = 2
        for i in range(tries):
            try:
                # NOTE: batch will be a torch tensor, so this sample function behaves
                # differently than sample from ReplayBuffer, which returns np arrays
                # we still torchify batch in agent to change the dtypes to float32 and move to gpu
                ptrs, batch = next(self.iter)
                state, action, next_state, reward, done = batch
                reward = reward.unsqueeze(-1)
                done = done.unsqueeze(-1)

                if self.sample_mode == 'transition' and len(ptrs) != batch_size:
                    raise RuntimeError('wrong batch size sampled')

                batch = (state, action, next_state, reward, done)
                ptrs = ptrs.cpu().numpy()

                if self.set_context_with_batch:
                    context = state['achieved_goal'].detach().clone()

                    if self.T_context is not None:
                        selected_t = sample_context_indices(
                            context.shape[0], self.T_context, sample_sides=True
                        )
                        context = context[selected_t, ...]

                    context = context.unsqueeze(0)

                    # subsampling after setting context with batch
                    # to allow context to be sampled from full trajectories
                    # if batch_size_opt too large and doesn't fit in gpu memory
                    if self.subsample_batch_size is not None:
                        N = reward.shape[0]
                        t = np.rng.choice(
                            np.arange(N),
                            size=self.subsample_batch_size,
                            replace=False,
                            shuffle=False,
                        )
                        t = np.sort(t)

                        state = {k: v[t, ...] for k, v in state.items()}
                        action = action[t, ...]
                        next_state = {k: v[t, ...] for k, v in next_state.items()}
                        reward = reward[t, ...]
                        done = done[t, ...]

                    state['context_demo'] = context
                    batch = (state, action, next_state, reward, done)

                self.batches[batch_it] = batch  # cache
                return ptrs
            except StopIteration:
                self.iter = iter(self.dataloader)
                self.epoch += 1
        raise Exception('Failed to get batch')

    def get_batch(self, batch_it, ptrs):
        return ptrs, self.batches.pop(batch_it)

    def get_state(self, ptrs, k_=None):
        states = []
        for ptr in ptrs:
            ptr, transition = self.dataset[ptr]
            state, action, next_state, reward, done = transition
            states.append(state)

        if k_ is not None:
            return np.stack([x[k_] for x in states])
        else:
            # batch
            s = {}
            for k in states[0].keys():
                s[k] = np.stack([x[k] for x in states])
            return s
