from dsb.dependencies import *
from torch.utils.data import Dataset, DataLoader

from ..utils import worker_init_fn
from .buffer_wrapper import BufferWrapper


class TransitionBufferDataset(Dataset):
    def __init__(self, batches_ptrs, get_batch_func=None):
        self.batches_ptrs = batches_ptrs
        self.get_batch_func = get_batch_func

    def __len__(self):
        return len(self.batches_ptrs)

    def __getitem__(self, idx):
        ptrs = self.batches_ptrs[idx]
        ptrs, batch = self.get_batch_func(ptrs)
        return ptrs, batch


class DataLoaderBatchedBufferWrapper(BufferWrapper):
    """
    must use storage w/ use_shmem=True
    this is like 5-10% faster with GPU as dataloader uses pin_memory=True

    Not inheriting from SampleBufferWrapper since this should be the last BufferWrapper if used
    so can just directly override sample
    """

    def __init__(self, buffer, num_workers=0, load_batches=True, **kwargs):
        super().__init__(buffer, **kwargs)
        self.iter_batch_size = None
        self.num_workers = num_workers
        self.load_batches = load_batches

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     del state['dataset']
    #     del state['dataloader']
    #     del state['iter']
    #     del state['iter_batch_size']
    #     return state

    # def __setstate__(self, state):
    #     state['dataset'] = None
    #     state['dataloader'] = None
    #     state['iter'] = None
    #     state['iter_batch_size'] = None
    #     self.__dict__.update(state)

    def ready_batches(self, num_batches, batch_size):
        self.batches_ptrs = self.sample_ptrs(num_batches * batch_size)
        if self.load_batches:
            self.batches_ptrs = self.batches_ptrs.reshape(num_batches, batch_size)

        self.dataset = TransitionBufferDataset(
            self.batches_ptrs, get_batch_func=self.buffer.get_batch
        )
        self.dataloader = DataLoader(
            self.dataset,
            pin_memory=torch.cuda.is_available(),
            shuffle=False,  # don't need to shuffle since ptrs should already be randomly ordered
            batch_size=None if self.load_batches else batch_size,
            worker_init_fn=worker_init_fn if self.num_workers else None,
            num_workers=self.num_workers,
        )

        self.iter = iter(self.dataloader)
        self.iter_batch_size = batch_size
        self.iter_counter = 0

    def cleanup_batches(self):
        if self.load_batches:
            assert self.iter_counter == len(self.batches_ptrs)
        else:
            assert self.iter_counter == len(self.batches_ptrs) / self.iter_batch_size
        self.batches_ptrs = None
        self.dataset = None
        self.dataloader = None
        self.iter = None
        self.iter_batch_size = None
        self.iter_counter = None

    # override buffer.sample so using dataloader iter instead
    def sample(self, batch_it, batch_size):
        assert batch_size == self.iter_batch_size
        assert batch_it == self.iter_counter

        # can just grab ptrs directly b/c not shuffling
        if self.load_batches:
            ptrs = self.batches_ptrs[batch_it]
        else:
            ptrs = self.batches_ptrs[batch_it * batch_size : (batch_it + 1) * batch_size]

        # NOTE: batch will be a torch tensor, so this sample function
        # behaves differently than sample from ReplayBuffer, which returns np arrays
        # we still torchify batch to change the dtypes to float32 and move to gpu
        # _ptrs, batch = next(self.iter)
        batch = next(self.iter)
        self.iter_counter += 1

        # print(ptrs[0], _ptrs[0])

        state, action, next_state, reward, done = batch
        # reward = reward.unsqueeze(-1)
        # done = done.unsqueeze(-1)
        # # don't need to reshape these since dataloader is not batching

        assert len(ptrs) == batch_size

        batch = (state, action, next_state, reward, done)
        return ptrs, batch
