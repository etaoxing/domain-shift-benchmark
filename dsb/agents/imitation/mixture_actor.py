from dsb.dependencies import *

from ..concat_state import concat_state


# from https://github.com/SudeepDasari/one_shot_transformers/blob/ecd43b0c182451b67219fdbc7d6a3cd912395f17/hem/models/inverse_module.py#L106
# They report "For most of our experiments, the model performed best when using two mixture
# components and learned constant variance parameters per action dimension", so n_mixtures=2 and const_var=True
# also see https://github.com/ikostrikov/jaxrl/blob/8ac614b0c5202acb7bb62cdb1b082b00f257b08c/jaxrl/networks/policies.py#L47
@concat_state
class MixtureActor(nn.Module):
    def __init__(self, mdp_space, n_mixtures=3, const_var=False, hidden_dim=256):
        super().__init__()
        in_dim = mdp_space.state_dim
        out_dim = mdp_space.action_dim
        max_action = mdp_space.max_action

        self.max_action = max_action  # if not 1, then need to scale output
        assert n_mixtures >= 1, "must predict at least one mixture!"
        self._n_mixtures, self._dist_size = n_mixtures, torch.Size((out_dim, n_mixtures))

        self._l = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self._mu = nn.Linear(hidden_dim, out_dim * n_mixtures)
        self._const_var = const_var

        if const_var:
            # independent of state, still optimized
            ln_scale = torch.randn(out_dim, dtype=torch.float32) / np.sqrt(out_dim)
            self.register_parameter('_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
        else:
            # state dependent
            self._ln_scale = nn.Linear(hidden_dim, out_dim * n_mixtures)
        self._logit_prob = nn.Linear(hidden_dim, out_dim * n_mixtures) if n_mixtures > 1 else None

    def forward(self, x):
        x = self._l(x)

        mu = self._mu(x).reshape((x.shape[:-1] + self._dist_size))
        if self._const_var:
            ln_scale = self._ln_scale if self.training else self._ln_scale.detach()
            ln_scale = ln_scale.reshape((1, -1, 1)).expand_as(mu)
            # NOTE: changed from (1, 1, -1, 1) to (1, -1, 1) b/c we don't have T dimension
        else:
            ln_scale = self._ln_scale(x).reshape((x.shape[:-1] + self._dist_size))

        logit_prob = (
            self._logit_prob(x).reshape((x.shape[:-1] + self._dist_size))
            if self._n_mixtures > 1
            else torch.ones_like(mu)
        )
        return (mu, ln_scale, logit_prob)
