from dsb.dependencies import *

from .buffer_wrapper import SampleBufferWrapper
from .storage import BaseStorage


class ReplayBuffer(SampleBufferWrapper):
    def __init__(self, storage, **unused):
        super().__init__(storage)

        if not isinstance(self.buffer, BaseStorage):
            raise RuntimeError

        self.storage = storage

    # @property
    # def ptr(self):
    #     return self.buffer.ptr

    def add(self, *transition, last_step=False, env_idx=None):  # doesn't have to be process safe
        ptr = self.buffer.add_directly(*transition, last_step=last_step)
        return ptr

    def sample_ptrs(self, batch_it, batch_size):
        try:
            ptrs = np.rng.integers(0, self.buffer.size, size=batch_size)
        except ValueError as e:
            raise ValueError(f'Need more samples in buffer, {e}')
        return ptrs

    def get_batch(self, batch_it, ptrs):
        return self.storage.get_batch(ptrs)
