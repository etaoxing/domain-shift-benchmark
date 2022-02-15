from dsb.dependencies import *
from dsb.utils import module_repr_include


_STATE_KEYS = {
    'goalconditioned': ('achieved_goal', 'desired_goal', 'observation'),
}


def get_state_keys_and_mdp_space(state_keys, full_mdp_space, skip_concat_warning=False):
    if state_keys is not None:
        if isinstance(state_keys, str):
            state_keys = _STATE_KEYS[state_keys]

        mdp_space = copy.deepcopy(full_mdp_space)
        mdp_space.state_keys = copy.deepcopy(state_keys)
    else:
        if not skip_concat_warning:
            raise RuntimeError(
                'Currently only supporting gym.Dict obs spaces. If obs space is gym.Dict '
                'then probably need to specifiy state_keys to concat'
            )
        mdp_space = copy.deepcopy(full_mdp_space)
        mdp_space.state_keys = None

    return state_keys, mdp_space


def cat_state(state, state_keys, concat_fn):
    if state_keys is not None:
        if concat_fn == 'torch':
            x = torch.cat([state[k] for k in state_keys], dim=-1)
        elif concat_fn == 'numpy':
            x = np.concatenate([state[k] for k in state_keys], axis=-1)
    else:
        x = state
    return x


# We use this decorator to wrap nn.Module so that they may accept
# a dictionary of tensors as input.
# Tensors in the input dict are selected using state_keys and concatenated
# (so they should all be 1-dimensional tensors).
def concat_state(ModuleCls):
    class ConcatStateModule(ModuleCls):
        def __init__(
            self,
            full_mdp_space,
            state_keys=None,
            concat_fn='torch',
            concat_next_state=False,
            skip_concat_warning=False,
            **kwargs
        ):
            self.concat_fn = concat_fn
            self.concat_next_state = concat_next_state

            self.state_keys, mdp_space = get_state_keys_and_mdp_space(
                state_keys, full_mdp_space, skip_concat_warning=skip_concat_warning
            )
            super().__init__(mdp_space, **kwargs)

            self.__name__ = ModuleCls.__name__

        def forward(self, state, *args, **kwargs):
            x = cat_state(state, self.state_keys, self.concat_fn)

            if self.concat_next_state:
                assert len(args) == 1
                next_state = args[0]
                x2 = cat_state(next_state, self.state_keys, self.concat_fn)
                args = (x2,)

            return super().forward(x, *args, **kwargs)

        def __repr__(self):
            s = module_repr_include(
                super().__repr__(),
                dict(
                    state_keys=self.state_keys,
                ),
            )
            return s

    # override this b/c used by builder.registry()
    ConcatStateModule.__name__ = ModuleCls.__name__

    return ConcatStateModule
