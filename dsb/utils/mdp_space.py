from dsb.dependencies import *


class MDPSpace:
    # holds obs space and action space
    # helps compute dimensions of inputs
    def __init__(
        self,
        obs_space,
        action_space,
        embedding_dim=None,
        embedding_keys=[],
    ):
        self.obs_space = obs_space
        self.action_space = action_space

        self.embedding_dim = embedding_dim
        self.embedding_keys = embedding_keys

        self.state_keys = []  # set later by ConcatStateModule

    @property
    def state_keys(self):
        return self._state_keys

    @state_keys.setter
    def state_keys(self, v):
        self._state_keys = v
        self.state_dim = self.compute_state_dim(v)

    def compute_state_dim(self, state_keys):
        if state_keys is None:
            return self.embedding_dim
        else:
            state_dim = 0
            for k in state_keys:
                if k in self.embedding_keys:
                    state_dim += self.embedding_dim
                else:
                    s = self.obs_space[k]
                    if len(s.shape) != 1:
                        raise RuntimeError
                    state_dim += s.shape[0]
            return state_dim

    @property
    def action_space(self):
        return self._action_space

    @action_space.setter
    def action_space(self, action_space):
        self._action_space = action_space

        if isinstance(action_space, gym.spaces.Box):
            self.action_dim = action_space.shape[0]
            self.max_action = float(action_space.high[0])
            self.action_type = 'continuous'
        elif isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n
            self.max_action = None
            self.action_type = 'discrete'
        else:
            raise RuntimeError

    def __repr__(self):
        s = 'MDPSpace('
        try:
            s += f'\n  (obs_space): {str(self.obs_space)}'
        except ValueError:
            # ValueError: zero-size array to reduction operation minimum which has no identity
            pass

        s += f'\n  (embedding_dim): {self.embedding_dim}'
        s += f'\n  (action_dim): {self.action_dim}'
        s += '\n)'
        return s
