from dsb.dependencies import *
import dsb.builder as builder
from dsb.agents.concat_state import concat_state
from dsb.agents.utils import variance_initializer_


@concat_state
class InverseModel(nn.Module):
    def __init__(self, mdp_space, hidden_dim=256):
        super().__init__()
        state_dim = mdp_space.state_dim
        action_dim = mdp_space.action_dim
        max_action = mdp_space.max_action

        self.l1 = nn.Linear(state_dim * 2, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.reset_parameters()

    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        a = torch.tanh(self.l3(x))
        a = self.max_action * a
        return a

    def reset_parameters(self):
        # nn.init.kaiming_uniform_(self.l1.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.zeros_(self.l1.bias)
        # nn.init.kaiming_uniform_(self.l2.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.zeros_(self.l2.bias)
        # nn.init.uniform_(self.l3.weight, -0.003, 0.003)
        # nn.init.zeros_(self.l3.bias)

        variance_initializer_(
            self.l1.weight, scale=1.0 / 3.0, mode='fan_in', distribution='uniform'
        )
        nn.init.zeros_(self.l1.bias)
        variance_initializer_(
            self.l2.weight, scale=1.0 / 3.0, mode='fan_in', distribution='uniform'
        )
        nn.init.zeros_(self.l2.bias)
        nn.init.uniform_(self.l3.weight, -0.003, 0.003)
        nn.init.zeros_(self.l3.bias)


from dsb.agents.imitation.mixture_actor import MixtureActor
from dsb.agents.imitation.behavioral_cloning import __DistributionCls__


# https://github.com/SudeepDasari/one_shot_transformers/blob/ecd43b0c182451b67219fdbc7d6a3cd912395f17/hem/models/inverse_module.py#L139
# https://github.com/SudeepDasari/one_shot_transformers/blob/ecd43b0c182451b67219fdbc7d6a3cd912395f17/hem/models/inverse_module.py#L164
# https://github.com/SudeepDasari/one_shot_transformers/blob/ecd43b0c182451b67219fdbc7d6a3cd912395f17/scripts/train_transformer.py#L34
@concat_state
class InverseMixtureModel(nn.Module):
    def __init__(
        self,
        mdp_space,
        mix_dist_params=dict(cls='DiscretizedLogisticMixture', num_bins=256),
        hidden_dim=256,
        model_params={},
    ):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(mdp_space.state_dim * 2, hidden_dim))

        modified_mdp_space = copy.deepcopy(mdp_space)
        modified_mdp_space.embedding_dim = hidden_dim
        self.model = MixtureActor(modified_mdp_space, skip_concat_warning=True, **model_params)

        c, self.mix_dist_params = builder.parse_params(mix_dist_params)
        self.DistributionCls = __DistributionCls__[c]

    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=1)
        x = self.fc(x)
        return self.model(x)


__InverseModelCls__ = builder.registry([InverseModel, InverseMixtureModel])
__ForwardModelCls__ = builder.registry([])
__ObsModelCls__ = builder.registry([])


class BasicDynamics(nn.Module):
    def __init__(
        self,
        mdp_space,
        optimize_interval=1,
        optim_params=dict(cls='Adam', lr=1e-3),
        inverse_model_params=None,
        forward_model_params=None,
        obs_model_params=None,
        embedding_head_param_group=None,
    ):
        super().__init__()
        self.mdp_space = mdp_space
        self.optimize_interval = optimize_interval

        if inverse_model_params is not None:
            self.inverse_model = builder.build_module(
                __InverseModelCls__, inverse_model_params, mdp_space
            )
        else:
            self.inverse_model = None

        if forward_model_params is not None:
            self.forward_model = builder.builder_module(
                __ForwardModelCls__, forward_model_params, mdp_space
            )
        else:
            self.forward_model = None

        if obs_model_params is not None:
            self.obs_model = builder.build_module(__ObsModelCls__, obs_model_params, mdp_space)
        else:
            self.obs_model = None

        self.optimizer = builder.build_optim(optim_params, params=self.parameters())
        if embedding_head_param_group:
            self.optimizer.add_param_group(embedding_head_param_group)

    def reset_parameters(self):
        if self.inverse_model:
            self.inverse_model.reset_parameters()
        if self.forward_model:
            self.forward_model.reset_parameters()
        if self.obs_model:
            self.obs_model.reset_parameters()

    def forward(self, state, next_state, action, deterministic=None):
        o = {}
        if self.inverse_model:
            if hasattr(self.inverse_model, 'mix_dist_params'):
                mu_bc, scale_bc, logit_bc = self.inverse_model(state, next_state)
                action_dist = self.inverse_model.DistributionCls(
                    mu_bc, scale_bc, logit_bc, **self.inverse_model.mix_dist_params
                )
                # TODO: multiply by max_action?
                if deterministic:
                    o_inv = action_dist.mean
                else:
                    o_inv = action_dist.sample()
                o['inverse'] = o_inv
            else:
                o['inverse'] = self.inverse_model(state, next_state)
        if self.forward_model:
            o['forward'] = self.forward_model(state, action)
        if self.obs_model:
            o['obs'] = self.obs_model(state['achieved_goal'])
        return o

    def optimize(self, state, next_state, action):
        opt_info = {}

        loss = 0
        if self.inverse_model:
            inverse_loss = self._inverse_loss(state, next_state, action)
            loss += inverse_loss
            opt_info['dynamics_inverse_loss'] = inverse_loss.item()

        if self.forward_model:
            forward_loss = self._forward_loss(state, next_state, action)
            loss += forward_loss
            opt_info['dynamics_forward_loss'] = forward_loss.item()

        if self.obs_model:
            obs_loss = self._obs_loss(state['achieved_goal'], state['observation'])
            loss += obs_loss
            opt_info['dynamics_obs_loss'] = obs_loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        opt_info['dynamics_total_loss'] = loss.item()
        return opt_info

    # predict a_t given s_t, s_t+1
    def _inverse_loss(self, state, next_state, action):
        if hasattr(self.inverse_model, 'mix_dist_params'):
            mu_bc, scale_bc, logit_bc = self.inverse_model(state, next_state)
            action_dist = self.inverse_model.DistributionCls(
                mu_bc, scale_bc, logit_bc, **self.inverse_model.mix_dist_params
            )
            return torch.mean(-action_dist.log_prob(action))
        else:
            pred_action = self.inverse_model(state, next_state)
            return F.mse_loss(pred_action, action)

    # predict s_t+1 given s_t, a_t
    def _forward_loss(self):
        raise NotImplementedError

    # predict proprioceptive state given (image/video) s_t
    def _obs_loss(self):
        raise NotImplementedError

    @property
    def opt_info_keys(self):
        k = ['dynamics_total_loss']
        if self.inverse_model:
            k.append('dynamics_inverse_loss')
        if self.forward_model:
            k.append('dynamics_forward_loss')
        if self.obs_model:
            k.append('dynamics_obs_loss')
        return k
