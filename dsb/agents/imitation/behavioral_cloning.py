from dsb.dependencies import *
import dsb.builder as builder
from torch.distributions import Distribution
import torch.distributions as D


class NormalMixture(Distribution):
    def __init__(self, mean, log_scale, logit_probs, temperature=1.0):
        batch_shape = log_scale.shape[:-1]
        event_shape = mean.shape[len(batch_shape) + 1 :]
        super().__init__(batch_shape, event_shape, None)

        mix = D.Categorical(logits=logit_probs)
        comp = D.Normal(mean, torch.exp(log_scale) * temperature)
        self.dist = D.MixtureSameFamily(mix, comp)
        # TODO: D.Independent?
        # see https://github.com/ikostrikov/jaxrl/blob/8ac614b0c5202acb7bb62cdb1b082b00f257b08c/jaxrl/networks/policies.py#L47
        # https://pytorch.org/docs/stable/distributions.html#mixturesamefamily

        # TODO: sigmoid or tanh when gaussianpolicy?
        # https://github.com/rail-berkeley/rlkit/blob/354f14c707cc4eb7ed876215dd6235c6b30a2e2b/rlkit/torch/sac/policies/gaussian_policy.py#L132

    def log_prob(self, value):
        log_probs = self.dist.log_prob(value)
        assert log_probs.shape == value.shape
        return log_probs

    def sample(self):
        return self.dist.sample()

    @property
    def mean(self):
        return self.dist.mean


from .discretized_logistic_mixture import DiscretizedLogisticMixture
from .mixture_actor import MixtureActor

__ActorCls__ = builder.registry([MixtureActor])
__DistributionCls__ = builder.registry([DiscretizedLogisticMixture, NormalMixture])


# maximum-likelihood with (mixture) policy parameterized by some distribution
# also see https://github.com/rail-berkeley/rlkit/blob/354f14c707cc4eb7ed876215dd6235c6b30a2e2b/rlkit/torch/distributions.py
class BehavioralCloning(nn.Module):
    def __init__(
        self,
        mdp_space,
        actor_params=dict(cls='MixtureActor'),
        mix_dist_params=dict(cls='DiscretizedLogisticMixture', num_bins=256),
        optimize_interval=1,
        actor_optim_params=dict(cls='Adam', lr=1e-3),
        embedding_head_param_group=None,
    ):
        super().__init__()
        self.mdp_space = mdp_space
        self.optimize_interval = optimize_interval

        self.actor = builder.build_module(__ActorCls__, actor_params, mdp_space)
        self.actor_optimizer = builder.build_optim(
            actor_optim_params, params=self.actor.parameters()
        )
        if embedding_head_param_group:
            self.actor_optimizer.add_param_group(embedding_head_param_group)

        c, self.mix_dist_params = builder.parse_params(mix_dist_params)
        self.DistributionCls = __DistributionCls__[c]

    def forward(self, state, deterministic=None):
        mu_bc, scale_bc, logit_bc = self.actor(state)
        action_dist = self.DistributionCls(mu_bc, scale_bc, logit_bc, **self.mix_dist_params)
        if deterministic:
            return action_dist.mean * self.actor.max_action
        else:
            return action_dist.sample() * self.actor.max_action

    def optimize(self, state, action):
        opt_info = {}

        mu_bc, scale_bc, logit_bc = self.actor(state)
        action_dist = self.DistributionCls(mu_bc, scale_bc, logit_bc, **self.mix_dist_params)

        bc_loss = torch.mean(-action_dist.log_prob(action))
        opt_info['bc_loss'] = bc_loss.item()

        self.actor_optimizer.zero_grad()
        bc_loss.backward()
        self.actor_optimizer.step()

        return opt_info

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['actor_optimizer'] = self.actor_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.actor_optimizer.load_state_dict(state_dict.pop('actor_optimizer'))
        super().load_state_dict(state_dict)

    @property
    def opt_info_keys(self):
        return ['bc_loss']
