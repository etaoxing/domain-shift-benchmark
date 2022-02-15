from dsb.dependencies import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from .sample_buffer_wrapper import SampleBufferWrapper


class CollateTrajectoryBufferWrapper(SampleBufferWrapper):
    # given a batch (B, ...) of transitions
    # collates them into batch (B, T, ...) of trajectory windows using PaddedSequence
    # If the batch just contains a single trajectory
    # then just adding a dimension

    # If using mode='pad', then will convert batch to torch tensors

    def __init__(self, buffer, mode='pad', **kwargs):
        super().__init__(buffer, **kwargs)
        assert mode in ['single_trajectory', 'pad']
        self.mode = mode

    def get_batch(self, batch_it, ptrs):
        ptrs, batch = self.buffer.get_batch(batch_it, ptrs)
        state, action, next_state, reward, done = batch

        start_ptrs = self.start_ptr[ptrs]

        start_ptrs_set, _length = np.unique(start_ptrs, return_counts=True)

        # sort episode start ptrs by decreasing episode lengths
        _length, start_ptrs_set = zip(*sorted(zip(_length, start_ptrs_set), reverse=True))

        # batch comes from same trajectory, assume ordered by timestep
        if self.mode == 'single_trajectory':
            assert len(start_ptrs_set) == 1
            state = {k: v[np.newaxis, ...] for k, v in state.items()}
            action = action[np.newaxis, ...]
            next_state = {k: v[np.newaxis, ...] for k, v in next_state.items()}
            reward = reward[np.newaxis, ...]
            done = done[np.newaxis, ...]

            # TODO: add _length?
        else:
            _state = collections.defaultdict(list)
            _action = []
            _next_state = collections.defaultdict(list)
            _reward = []
            _done = []
            for p in start_ptrs_set:
                mask = start_ptrs == p

                for k, v in state.items():
                    _state[k].append(v[mask])
                _action.append(action[mask])
                for k, v in next_state.items():
                    _next_state[k].append(v[mask])
                _reward.append(reward[mask])
                _done.append(done[mask])

            state = {k: pad_sequence(v, batch_first=True) for k, v in _state.items()}
            action = pad_sequence(_action, batch_first=True)
            next_state = {k: pad_sequence(v, batch_first=True) for k, v in _next_state.items()}
            reward = pad_sequence(_reward, batch_first=True)
            done = pad_sequence(_done, batch_first=True)

            state['_length'] = _length

        batch = (state, action, next_state, reward, done)
        return ptrs, batch


class UnCollateTrajectoryBufferWrapper(SampleBufferWrapper):
    def __init__(self, buffer, mode='pad', **kwargs):
        super().__init__(buffer, **kwargs)
        assert mode in ['single_trajectory', 'pad']
        self.mode = mode

    def get_batch(self, batch_it, ptrs):
        ptrs, batch = self.buffer.get_batch(batch_it, ptrs)
        state, action, next_state, reward, done = batch

        if self.mode == 'single_trajectory':
            assert done.shape[0] == 1

            state = {k: v[0, ...] for k, v in state.items()}
            action = action[0, ...]
            next_state = {k: v[0, ...] for k, v in next_state.items()}
            reward = reward[0, ...]
            done = done[0, ...]
        else:
            _length = state.pop('_length')

            _state = collections.defaultdict(list)
            _action = []
            _next_state = collections.defaultdict(list)
            _reward = []
            _done = []
            for i, N in enumerate(_length):
                for k, v in state.items():
                    _state[k].append(v[i, :N, ...])
                _action.append(action[i, :N, ...])
                for k, v in next_state.items():
                    _next_state[k].append(v[i, :N, ...])
                _reward.append(reward[i, :N, ...])
                _done.append(done[i, :N, ...])

            state = {k: torch.cat(v) for k, v in _state.items()}
            action = torch.cat(_action)
            next_state = {k: torch.cat(v) for k, v in _next_state.items()}
            reward = torch.cat(_reward)
            done = torch.cat(_done)

        batch = (state, action, next_state, reward, done)
        return ptrs, batch


def pack_padded_batch(_length, batch, unpack=False):
    state, action, next_state, reward, done = batch

    if not unpack:
        state = {k: pack_padded_sequence(v, _length, batch_first=True) for k, v in state.items()}
        action = pack_padded_sequence(action, _length, batch_first=True)
        next_state = {
            k: pack_padded_sequence(v, _length, batch_first=True) for k, v in next_state.items()
        }
        reward = pack_padded_sequence(reward, _length, batch_first=True)
        done = pack_padded_sequence(done, _length, batch_first=True)
    else:
        state = {k: pad_packed_sequence(v, _length, batch_first=True) for k, v in state.items()}
        action = pad_packed_sequence(action, _length, batch_first=True)
        next_state = {
            k: pad_packed_sequence(v, _length, batch_first=True) for k, v in next_state.items()
        }
        reward = pad_packed_sequence(reward, _length, batch_first=True)
        done = pad_packed_sequence(done, _length, batch_first=True)

    batch = (state, action, next_state, reward, done)
    return batch


class PackPaddedBufferWrapper(SampleBufferWrapper):
    # Turn PaddedSequence into PackedPaddedSequence

    def __init__(self, buffer, **kwargs):
        super().__init__(buffer, **kwargs)

    def get_batch(self, batch_it, ptrs):
        ptrs, batch = self.buffer.get_batch(batch_it, ptrs)
        state, action, next_state, reward, done = batch

        _length = state.pop('_length')

        batch = pack_padded_batch(_length, batch)

        state['_length'] = _length

        batch = (state, action, next_state, reward, done)
        return ptrs, batch
