from dsb.dependencies import *
from ..utils import get_pair


class EmbeddingHeadMixin:
    def compute_embedding(
        self,
        _state,
        state_embedding=None,
        use_target_embedding_head=False,
        **kwargs,
    ):
        state = {}  # TODO: if inplace=True, swap back into _state
        for k, v in _state.items():
            # contains already computed embeddings
            if state_embedding is not None and k in state_embedding.keys():
                pass
            else:
                state[k] = v

        if len(state.keys()) > 0:
            if self.state_normalizer:
                # NOTE: currently, state normalizer creates a copy of a dict, so inplace=True does not work as intended.
                # this is fine for now b/c we don't want to change the state data in agent.optimize
                state = self.state_normalizer(state)

            if use_target_embedding_head:
                assert self.embedding_head_target is not None
                state = self.embedding_head_target(state, **kwargs)
            else:
                state = self.embedding_head(state, **kwargs)

            if state_embedding is not None:
                for k, v in state_embedding.items():
                    state[k] = v  # substitute embeddings
        else:
            state = state_embedding

        return state

    def get_embedding_target(self, buffer, ptrs, state, next_state):
        pair_type = getattr(self.embedding_head, 'pair_type', None)

        if (
            hasattr(self.state_normalizer, 'normalizers')
            and 'embedding_target' in self.state_normalizer.normalizers.keys()
        ):  # for RAE reconstruction target
            unnormalized_ag = state['desired_goal']
            # unnormalized_ag = self.state_normalizer.normalizers['desired_goal'](state['desired_goal'], inv_norm=True)
            embedding_target = self.state_normalizer(unnormalized_ag, which='embedding_target')

            state = self.state_normalizer(state)
        elif pair_type is None:
            embedding_target = None
            if self.state_normalizer:
                state = self.state_normalizer(state)
        elif pair_type == 'state':
            embedding_target = None
            if self.state_normalizer:
                state = self.state_normalizer(state)
        elif pair_type == 'sampled_pair':
            state_pair, distance_pair, pos_mask = get_pair(buffer, ptrs)
            state_pair = state_pair['desired_goal']

            if self.state_normalizer:
                state_pair = self.state_normalizer(state_pair, which='desired_goal')
                state = self.state_normalizer(state)
            embedding_target = (state_pair, distance_pair, pos_mask)
        else:
            raise ValueError(f'Unsupported embedding_target pair_type: {pair_type}')

        return state, embedding_target

    def optimize_embedding_head(self, buffer, ptrs, batch):
        if self.optimize_iterations % self.embedding_head.optimize_interval == 0:
            _state, action, _next_state, reward, done = batch

            # NOTE: optimizing embedding on state won't include terminal state of episode in next_state
            # use next_state? https://github.com/vitchyr/rlkit/blob/ae49265ef049c6b6d6be4fd3b2c76c5131c9b0ce/rlkit/torch/vae/vae_trainer.py#L331

            with torch.no_grad():
                state, embedding_target = self.get_embedding_target(
                    buffer, ptrs, _state, _next_state
                )

            # NOTE: if embedding_head uses data augmentation, then we're performing
            # input normalization before that aug occurs. for images, things should be
            # fine if we use PixelRescaleNormalizer to have images in [0,1] range
            opt_info = self.embedding_head.optimize(state, embedding_target=embedding_target)
        else:
            opt_info = {k: None for k in self.embedding_head.opt_info_keys}
        return opt_info
