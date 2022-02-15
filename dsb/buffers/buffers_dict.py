from dsb.dependencies import *
import dsb.builder as builder
from dsb.utils import torchify

from .buffer_wrapper import SampleBufferWrapper
from .storage import BaseStorage


def _collate(x_list):
    if len(x_list) == 1:
        return x_list[0]

    if isinstance(x_list[0], dict):
        x = {}
        for k in x_list[0].keys():
            x[k] = _collate([x_list[i][k] for i in range(len(x_list))])
    else:
        if all(isinstance(x, np.ndarray) for x in x_list):
            x = np.concatenate(x_list)
        elif any(isinstance(x, torch.Tensor) for x in x_list):
            x = torch.cat([torchify(x) for x in x_list])
        else:
            print(x_list)
            raise RuntimeError
    return x


def collate_batch(b, k):
    num_batch_elements = len(b[0])
    num_to_collate = len(k)

    batch = [None] * num_batch_elements
    for i in range(num_batch_elements):
        x = _collate([b[j][i] for j in range(num_to_collate)])
        batch[i] = x
    return tuple(batch)


class BuffersDict(SampleBufferWrapper):
    def __init__(self, storage, batch_proportion=None, buffers_params={}, **kwargs):
        super().__init__(storage)

        if not isinstance(self.buffer, BaseStorage):
            raise RuntimeError

        if batch_proportion is not None:
            assert sum(batch_proportion.values()) == 1.0

        d = {}
        for k, v in buffers_params.items():
            if batch_proportion is not None:
                assert k in batch_proportion.keys()

            d[k] = builder.build_buffer(
                v, storage.obs_space, storage.action_space, tmp_dir=storage.tmp_dir, **kwargs
            )

        self.buffers_dict = d
        self.batch_proportion = batch_proportion

    def sample_ptrs(self, batch_it, batch_size):
        # TODO: check buffer.sample_mode,
        # and convert b/w batch_size of transitions and batch_size of trajectories

        if self.batch_proportion is None:
            bs = {k: batch_size for k in self.buffers_dict.keys()}
        else:
            # if some buffer_size==0, then will reweight proportions excluding that buffer
            s = 0.0
            for k, v in self.buffers_dict.items():
                if v.size == 0:
                    continue
                s += self.batch_proportion[k]

            bs = {
                k: int((self.batch_proportion[k] / s) * batch_size)
                for k in self.buffers_dict.keys()
            }

        o = {}
        for k, buffer in self.buffers_dict.items():
            if buffer.size == 0:
                continue

            batch_size_opt = bs[k]

            if hasattr(buffer, 'set_batch_size_opt'):
                buffer.set_batch_size_opt(batch_size_opt)

            o[k] = buffer.sample_ptrs(batch_it, batch_size_opt)
        return o

    def get_batch(self, batch_it, ptrs):
        p = []
        b = []
        k = []
        for _k, _v in ptrs.items():
            _ptrs, _batch = self.buffers_dict[_k].get_batch(batch_it, _v)
            p.append(_ptrs)
            b.append(_batch)
            k.append(_k)
        ptrs = np.concatenate(p)
        batch = collate_batch(b, k)
        return ptrs, batch

    def add(self, *args, **kwargs):
        return self.buffers_dict['replay'].add(*args, **kwargs)

    @property
    def datasets(self):
        return self.buffers_dict['demo'].datasets

    def keys(self):
        return self.buffers_dict.keys()

    def __getitem__(self, key):
        return self.buffers_dict[key]

    def __repr__(self):
        return self.buffers_dict.__repr__()
