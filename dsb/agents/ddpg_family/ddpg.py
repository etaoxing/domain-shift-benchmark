from dsb.dependencies import *
import dsb.builder as builder
from dsb.utils import torchify, untorchify, stop_grad

from ..concat_state import concat_state
from ..utils import variance_initializer_, update_target_network


# Returns an action for a given state
@concat_state
class Actor(nn.Module):
    def __init__(self, mdp_space, hidden_dim=256):
        super().__init__()
        state_dim = mdp_space.state_dim
        action_dim = mdp_space.action_dim
        max_action = mdp_space.max_action

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.reset_parameters()

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
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


# Returns a Q-value for given state/action pair
@concat_state
class Critic(nn.Module):
    def __init__(self, mdp_space, hidden_dim=256, output_dim=1, td3_style=False, clip_q=False):
        super().__init__()
        state_dim = mdp_space.state_dim
        action_dim = mdp_space.action_dim

        self.output_dim = output_dim
        self.td3_style = td3_style
        self.clip_q = clip_q

        if self.td3_style:
            self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, hidden_dim)
            self.l3 = nn.Linear(hidden_dim, output_dim)
        else:
            self.l1 = nn.Linear(state_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
            self.l3 = nn.Linear(hidden_dim, output_dim)

        self.reset_parameters()

    @property
    def is_distributional(self):
        return False if self.output_dim == 1 else True

    @property
    def num_bins(self):
        return self.output_dim

    def forward(self, state, action, **kwargs):
        if self.td3_style:
            sa = torch.cat([state, action], dim=1)
            q = F.relu(self.l1(sa))
            q = F.relu(self.l2(q))
            q = self.l3(q)
        else:
            q = F.relu(self.l1(state))
            q = F.relu(self.l2(torch.cat([q, action], dim=1)))
            q = self.l3(q)
        return q

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


__ModuleCls__ = builder.registry([Actor, Critic])


class DDPG(nn.Module):
    def __init__(
        self,
        mdp_space,
        discount=0.99,
        actor_optimize_interval=1,
        targets_update_interval=1,
        tau=0.005,
        action_l2=None,
        #
        target_policy_smoothing=False,
        policy_noise=0.2,
        noise_clip=0.5,
        #
        actor_detach_embedding=False,
        #
        actor_params=dict(cls='Actor'),
        critic_params=dict(cls='Critic'),
        actor_optim_params=dict(cls='Adam', lr=1e-3),
        critic_optim_params=dict(cls='Adam', lr=1e-3),
        embedding_head_param_group=None,
    ):
        super().__init__()
        self.mdp_space = mdp_space
        self.max_action = mdp_space.max_action

        self.discount = discount
        self.actor_optimize_interval = actor_optimize_interval
        self.targets_update_interval = targets_update_interval
        self.tau = tau
        self.action_l2 = action_l2

        # see TD3
        # https://github.com/sfujim/TD3/blob/7d5030587011a8bb285f457a75068d033e1365d0/TD3.py#L76
        self.target_policy_smoothing = target_policy_smoothing
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        self.actor_detach_embedding = actor_detach_embedding

        self.actor = builder.build_module(__ModuleCls__, actor_params, mdp_space)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
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

    def select_action(self, state, deterministic=None, **kwargs):
        return self.actor(state)

    def get_qvalues(self, state, actions=None):
        if actions is None:
            actions = self.actor(state)
        qvalues = self.critic(state, actions)
        return qvalues

    def critic_loss(self, current_q, target_q, reward, done, discount):
        td_targets = reward + ((1 - done) * discount * target_q).detach()
        critic_loss = F.mse_loss(current_q, td_targets)
        # critic_loss = F.smooth_l1_loss(current_q, td_targets) # Huber loss, if used then w/ distributional qvalues, actor_loss can diverge to max value
        return critic_loss

    def optimize(self, optimize_iterations, batch):
        opt_info = {}
        state, action, next_state, reward, done = batch

        with torch.no_grad():
            if self.target_policy_smoothing:
                # Select action according to policy and add clipped noise
                noise = (torch.randn_like(action) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )

                next_action = (self.actor_target(next_state) + noise).clamp(
                    -self.max_action, self.max_action
                )
            else:
                next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
        current_q = self.critic(state, action)
        critic_loss = self.critic_loss(current_q, target_q, reward, done, self.discount)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        opt_info['critic_loss'] = critic_loss.item()

        if optimize_iterations % self.actor_optimize_interval == 0:
            if self.actor_detach_embedding:
                state = stop_grad(state)
                # TODO: different target update rates

            # Compute actor loss
            actor_action = self.actor(state)
            qvalues = self.get_qvalues(state, actions=actor_action)
            actor_loss = -qvalues.mean()
            if self.action_l2 is not None:
                # l2 penalty on actions
                actor_loss += self.action_l2 * (actor_action / self.max_action).pow(2).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            opt_info['actor_loss'] = actor_loss.item()
            opt_info['qvalues'] = untorchify(qvalues.mean(dim=0))
        else:
            opt_info['actor_loss'] = None
            opt_info['qvalues'] = None

        # Update the frozen target models
        if optimize_iterations % self.targets_update_interval == 0:
            update_target_network(self.actor, self.actor_target, tau=self.tau)
            update_target_network(self.critic, self.critic_target, tau=self.tau)

        return opt_info

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['actor_optimizer'] = self.actor_optimizer.state_dict()
        state_dict['critic_optimizer'] = self.critic_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.actor_optimizer.load_state_dict(state_dict.pop('actor_optimizer'))
        self.critic_optimizer.load_state_dict(state_dict.pop('critic_optimizer'))
        super().load_state_dict(state_dict)
