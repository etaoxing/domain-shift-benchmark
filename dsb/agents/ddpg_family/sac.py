from dsb.dependencies import *
import dsb.builder as builder
from dsb.utils import torchify, untorchify, stop_grad

from ..concat_state import concat_state
from ..utils import variance_initializer_, update_target_network

# from https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/sac_ae.py#L186

from .ddpg import Critic
from .uvf_ddpg import EnsembledCritic


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def sac_weight_init(m):  # TODO: do this for embedding_head too
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


@concat_state
class SACActor(nn.Module):
    def __init__(
        self,
        mdp_space,
        hidden_dim=256,
        log_std_min=-10,
        log_std_max=2,
    ):
        super().__init__()
        state_dim = mdp_space.state_dim
        action_dim = mdp_space.action_dim
        max_action = mdp_space.max_action

        # https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/train.py#L166
        assert max_action == 1

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * action_dim),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(sac_weight_init)

    def forward(self, state, compute_pi=True, compute_log_pi=True):
        mu, log_std = self.trunk(state).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        # self.outputs['mu'] = mu
        # self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            # entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std
        # mu is the exploitation policy, pi is the exploration policy


class SACCritic(Critic):
    def __init__(
        self,
        *args,
        td3_style=True,  # https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/sac_ae.py#L124
        **kwargs,
    ):
        super().__init__(*args, td3_style=td3_style, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(sac_weight_init)


__ModuleCls__ = builder.registry([SACActor, SACCritic])


class SAC(nn.Module):
    def __init__(
        self,
        mdp_space,
        discount=0.99,
        actor_optimize_interval=2,
        targets_update_interval=2,  # critic and embedding head
        critic_tau=0.005,
        action_l2=None,
        init_temperature=0.01,
        critic_ensemble_size=2,
        critic_aggregate='min',
        max_q=None,
        actor_detach_embedding=True,
        actor_params=dict(cls='SACActor'),
        critic_params=dict(cls='SACCritic'),
        actor_optim_params=dict(cls='Adam', lr=1e-3),
        critic_optim_params=dict(cls='Adam', lr=1e-3),
        alpha_optim_params=dict(cls='Adam', lr=1e-3),
        embedding_head_param_group=None,
    ):
        super().__init__()
        self.mdp_space = mdp_space
        self.max_action = mdp_space.max_action
        action_dim = mdp_space.action_dim

        self.discount = discount
        self.actor_optimize_interval = actor_optimize_interval
        self.targets_update_interval = targets_update_interval
        self.critic_tau = critic_tau
        self.action_l2 = action_l2
        self.init_temperature = init_temperature

        self.max_q = max_q
        self.critic_ensemble_size = critic_ensemble_size
        self.critic_aggregate = critic_aggregate

        # SAC+AE stops actor gradients to conv encoder and uses different target update rates
        # https://arxiv.org/pdf/1910.01741.pdf#page=6
        # https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/sac_ae.py#L247
        # https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/sac_ae.py#L341
        self.actor_detach_embedding = actor_detach_embedding

        # action prior is gaussian, entropy is automatically tuned
        self.log_alpha = torch.tensor(np.log(init_temperature))
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        # action_dim is 1d, https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/sac_ae.py#L252
        self.target_entropy = -action_dim
        self.log_alpha_optimizer = builder.build_optim(alpha_optim_params, params=[self.log_alpha])

        self.actor = builder.build_module(__ModuleCls__, actor_params, mdp_space)
        self.actor_optimizer = builder.build_optim(
            actor_optim_params, params=self.actor.parameters()
        )
        if not self.actor_detach_embedding:
            if embedding_head_param_group:
                self.actor_optimizer.add_param_group(embedding_head_param_group)

        self.critic = builder.build_module(__ModuleCls__, critic_params, mdp_space)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = builder.build_optim(
            critic_optim_params, params=self.critic.parameters()
        )
        if embedding_head_param_group:
            self.critic_optimizer.add_param_group(embedding_head_param_group)

        if self.critic_ensemble_size > 1:
            self.critic = EnsembledCritic(
                self.critic, critic_ensemble_size=self.critic_ensemble_size
            )
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_target.load_state_dict(self.critic.state_dict())

            for i in range(1, len(self.critic.critics)):  # first copy already added
                critic_copy = self.critic.critics[i]
                self.critic_optimizer.add_param_group({'params': critic_copy.parameters()})
                # https://stackoverflow.com/questions/51756913/in-pytorch-how-do-you-use-add-param-group-with-a-optimizer

    @property
    def tau(self):
        return self.critic_tau

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=None, **kwargs):
        if deterministic:
            # should be used for eval
            mu, _, _, _ = self.actor(state, compute_pi=False, compute_log_pi=False)
            action = mu
        else:  # sample action
            # should be used for training
            mu, pi, _, _ = self.actor(state, compute_log_pi=False)
            action = pi
        return action

    def get_qvalues(self, state, actions=None, aggregate='default'):
        if aggregate == 'default':
            aggregate = self.critic_aggregate

        if actions is None:
            raise NotImplementedError
        qvalues = self.critic(state, actions)

        if self.critic_ensemble_size == 1:
            qvalues_list = [qvalues]
        else:
            qvalues_list = qvalues

        distributional = self.critic.is_distributional
        num_bins = self.critic.num_bins
        clip_q = self.critic.clip_q

        if distributional:
            raise NotImplementedError
        else:
            expected_qvalues_list = qvalues_list

        expected_qvalues = torch.stack(expected_qvalues_list)
        if aggregate is not None:
            if aggregate == 'min':
                expected_qvalues, _ = torch.min(expected_qvalues, dim=0)
            else:
                raise ValueError

        if not distributional and clip_q:
            # Clip the q values if not using distributional RL. If using
            # distributional RL, the q values are implicitly clipped.
            min_qvalue = -1.0 * self.max_q
            max_qvalue = 0.0
            expected_qvalues = torch.clamp(expected_qvalues, min_qvalue, max_qvalue)

        return expected_qvalues

    def critic_loss(self, log_pi, current_q, target_q, reward, done, discount):
        if not isinstance(current_q, list):
            current_q_list = [current_q]
            target_q_list = [target_q]
        else:
            current_q_list = current_q
            target_q_list = target_q

        distributional = self.critic.is_distributional
        num_bins = self.critic.num_bins
        clip_q = self.critic.clip_q

        if distributional:
            # TODO: https://arxiv.org/pdf/2004.14547.pdf
            # https://github.com/xtma/dsac/blob/master/rlkit/torch/dsac/dsac.py
            # https://github.com/xtma/dsac/blob/master/dsac.py
            # https://github.com/schatty/d4pg-pytorch/tree/master/models/d4pg

            # take the policy entropy, form a normal distribution, then
            # sample the log_prob for each bin?
            raise NotImplementedError
        else:
            assert self.critic_aggregate == 'min'

            critic_loss_list = []
            expected_qvalues, _ = torch.min(torch.stack(target_q_list), dim=0)

            if clip_q:
                # Clip the q values if not using distributional RL. If using
                # distributional RL, the q values are implicitly clipped.
                min_qvalue = -1.0 * self.max_q
                max_qvalue = 0.0
                expected_qvalues = torch.clamp(expected_qvalues, min_qvalue, max_qvalue)

            # NOTE: this is slightly different than in DDPG, where the
            # targets are computed per critic in ensemble. here, we're
            # computing the targets by aggregating over the ensemble
            target_v = expected_qvalues - self.alpha.detach() * log_pi
            td_targets = reward + ((1 - done) * discount * target_v).detach()
            for current_q in current_q_list:
                critic_loss = F.mse_loss(current_q, td_targets)
                critic_loss_list.append(critic_loss)
            # critic_loss = torch.sum(torch.stack(critic_loss_list))
            # NOTE: taking mean here, though SAC+AE sums, https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/sac_ae.py#L328
            critic_loss = torch.mean(torch.stack(critic_loss_list))
        return critic_loss

    def optimize(self, optimize_iterations, batch):
        opt_info = {}
        state, action, next_state, reward, done = batch

        # update_critic()
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_state)
            target_q = self.critic_target(next_state, policy_action)
        current_q = self.critic(state, action)
        critic_loss = self.critic_loss(log_pi, current_q, target_q, reward, done, self.discount)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        opt_info['critic_loss'] = critic_loss.item()

        if optimize_iterations % self.actor_optimize_interval == 0:
            if self.actor_detach_embedding:  # detach encoder
                state = stop_grad(state)

            _, pi, log_pi, log_std = self.actor(state)
            qvalues = self.get_qvalues(state, actions=pi)
            actor_loss = (self.alpha.detach() * log_pi - qvalues).mean()
            if self.action_l2 is not None:
                # l2 penalty on actions
                actor_loss += self.action_l2 * (pi / self.max_action).pow(2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            opt_info['actor_loss'] = actor_loss.item()
            opt_info['qvalues'] = untorchify(qvalues.mean(dim=0))
            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
            opt_info['entropy'] = entropy.mean().item()

            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            opt_info['alpha_loss'] = alpha_loss.item()
            opt_info['alpha'] = self.alpha.item()
        else:
            for k in ['actor_loss', 'qvalues', 'entropy', 'alpha_loss', 'alpha']:
                opt_info[k] = None

        # Update the frozen target models
        if optimize_iterations % self.targets_update_interval == 0:
            update_target_network(self.critic, self.critic_target, tau=self.critic_tau)

        return opt_info

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['log_alpha'] = self.log_alpha
        # state_dict['log_alpha_optimizer'] = self.log_alpha_optimizer.state_dict()
        state_dict['actor_optimizer'] = self.actor_optimizer.state_dict()
        state_dict['critic_optimizer'] = self.critic_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.log_alpha = state_dict.pop('log_alpha')  # should keep requires_grad property
        # self.log_alpha_optimizer.load_state_dict(state_dict.pop('log_alpha_optimizer'))
        self.actor_optimizer.load_state_dict(state_dict.pop('actor_optimizer'))
        self.critic_optimizer.load_state_dict(state_dict.pop('critic_optimizer'))
        super().load_state_dict(state_dict)
