from dsb.dependencies import *
from dsb.buffers.utils import get_timestep_of, sample_pos_pair_indices, sample_neg_pair_indices

from .buffer_wrapper import BufferWrapper


class SPTMPairSamplerBufferWrapper(BufferWrapper):
    """
    observations are considered close if within `l` timesteps.
    negative examples are pairs where obs are separated by `M * l` timesteps,
    where `M` is a margin b/w pos and neg examples. pairs are taken
    from the same episode trajectory

    in https://github.com/nsavinov/SPTM/blob/45592e5f86b3c509665e4d72c756633489a7124c/src/common/constants.py
    l = MAX_ACTION_DISTANCE
    M = NEGATIVE_SAMPLE_MULTIPLIER
    """

    def __init__(self, buffer, l=5, M=5, **kwargs):
        super().__init__(buffer, **kwargs)
        self.l = l
        self.M = M

    def sample_mask(self, batch_ptrs_anchor):
        # SPTM does a 50/50 split b/w pos/neg samples
        # https://github.com/nsavinov/SPTM/blob/45592e5f86b3c509665e4d72c756633489a7124c/src/train/train_edge_predictor.py#L23

        # HTM's implementation of SPTM is also doing an even split
        # however, for each anchor in the batch, HTM's version samples
        # 1 positive and 1 negative example, leading to 2 * batch_size pairs
        # https://github.com/thanard/hallucinative-topological-memory/blob/96ab39dba7ffd5363f26ea220607c454571076d9/trainer.py#L50
        # https://github.com/thanard/hallucinative-topological-memory/blob/82f63f01e7b6b552d515275249d5a11a5be6fe0a/models.py#L70

        batch_size = len(batch_ptrs_anchor)
        pos_mask = np.rng.uniform(size=batch_size) < 0.5
        return batch_ptrs_anchor, pos_mask

    def sample_pair_given(self, batch_ptrs_anchor):
        ptrs_anchor, pos_mask = self.sample_mask(batch_ptrs_anchor)

        neg_mask = ~pos_mask
        start_ptrs = self.start_ptr[ptrs_anchor]
        num_pairs = len(ptrs_anchor)

        ptrs_pair = np.ones(num_pairs, dtype=np.int) * -1
        distance_pair = np.ones(num_pairs, dtype=np.int) * -1

        for s in np.unique(start_ptrs):
            episode_ptrs = np.array(self.episode_ptrs[s][:])
            N = len(episode_ptrs)

            # TODO: if ptr is last timestep of episode, then must be marked as a neg example?
            # currently, just letting sample_pos_pair_indices pair it w/ itself
            # could also grab terminal state from next_state
            # SPTM does not use the last timestep, see https://github.com/nsavinov/SPTM/blob/45592e5f86b3c509665e4d72c756633489a7124c/src/train/train_edge_predictor.py#L26

            # positive examples
            episode_pos_mask = (start_ptrs == s) & (pos_mask)
            t_pos = get_timestep_of(ptrs_anchor[episode_pos_mask], episode_ptrs)
            episode_t_pos_pair = sample_pos_pair_indices(t_pos, N, 1, self.l)

            # negative examples
            episode_neg_mask = (start_ptrs == s) & (neg_mask)
            t_neg = get_timestep_of(ptrs_anchor[episode_neg_mask], episode_ptrs)

            episode_t_neg_pair = sample_neg_pair_indices(t_neg, N, self.M * self.l + 1)

            ptrs_pair[episode_pos_mask] = episode_ptrs[episode_t_pos_pair]
            ptrs_pair[episode_neg_mask] = episode_ptrs[episode_t_neg_pair]

            distance_pair[episode_pos_mask] = np.abs(episode_t_pos_pair - t_pos)
            distance_pair[episode_neg_mask] = np.abs(episode_t_neg_pair - t_neg)

        # TODO: should we be getting next_state instead as goal?
        state_pair = self.buffer.get_state(ptrs_pair, 'achieved_goal')
        state_pair = dict(desired_goal=state_pair)
        if self.buffer.framestack_timesteps is not None:
            state_pair['fs_desired_goal'] = self.buffer.get_state(ptrs_pair, 'fs_achieved_goal')
        # NOTE: taking achieved goals from trajectories and setting that as the desired goal for distance training

        if self.buffer.batch_dtype is not None:
            state_pair = {
                k: v.astype(self.buffer.batch_dtype, copy=True) for k, v in state_pair.items()
            }
        else:
            state_pair = {k: v.copy() for k, v in state_pair.items()}

        return ptrs_pair, state_pair, distance_pair, pos_mask


class HTMPairSamplerBufferWrapper(SPTMPairSamplerBufferWrapper):
    def __init__(
        self,
        buffer,
        num_negatives_per=50,  # https://github.com/thanard/hallucinative-topological-memory/blob/48182da80b53928647c4fb89178fb1e825f878a1/main.py#L52
        **kwargs
    ):
        super().__init__(buffer, **kwargs)
        self.num_negatives_per = num_negatives_per

    def sample_mask(self, batch_ptrs_anchor):
        batch_size = len(batch_ptrs_anchor)
        ptrs_all = np.tile(batch_ptrs_anchor, 1 + self.num_negatives_per)
        pos_mask = np.concatenate(
            [  # NOTE: ordering matters
                np.ones(batch_size, dtype=np.bool),
                np.zeros(batch_size * self.num_negatives_per, dtype=np.bool),
            ]
        )
        return ptrs_all, pos_mask


class CPCPairSamplerBufferWrapper(SPTMPairSamplerBufferWrapper):
    def __init__(self, buffer, **kwargs):
        super().__init__(buffer, **kwargs)

    def sample_mask(self, batch_ptrs_anchor):
        # uses other batch elements as negative samples rather than
        # explicitly sampling negative examples per anchor, so only sampling positives
        batch_size = len(batch_ptrs_anchor)
        pos_mask = np.ones(batch_size, dtype=np.bool)
        return batch_ptrs_anchor, pos_mask
