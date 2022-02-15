from dsb.dependencies import *
import dsb.builder as builder

from ..concat_state import concat_state
from ..utils import variance_initializer_


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


__ActorCls__ = builder.registry([Actor])


# see section 3.2 of https://arxiv.org/pdf/1811.06711.pdf for a discussion on BC surrogate losses
# optimizing by MSE yields deterministic policy
class MSEBehavioralCloning(nn.Module):
    def __init__(
        self,
        mdp_space,
        actor_params=dict(cls='Actor'),
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

    def forward(self, state, deterministic=None):
        return self.actor(state)

    def bc_loss(self, state, action):
        pred_action = self.actor(state)
        bc_loss = F.mse_loss(pred_action, action.detach())
        return bc_loss

    def optimize(self, state, action):
        opt_info = {}

        bc_loss = self.bc_loss(state, action)

        self.actor_optimizer.zero_grad()
        bc_loss.backward()
        self.actor_optimizer.step()

        opt_info['bc_loss'] = bc_loss.item()

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
