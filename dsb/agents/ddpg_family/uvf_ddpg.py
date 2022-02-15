from dsb.dependencies import *
import dsb.builder as builder

from .ddpg import DDPG


class EnsembledCritic(nn.Module):
    def __init__(self, CriticInstance, critic_ensemble_size=3):
        super().__init__()
        self.critic_ensemble_size = critic_ensemble_size

        self.critics = nn.ModuleList([CriticInstance])
        for _ in range(self.critic_ensemble_size - 1):
            critic_copy = copy.deepcopy(CriticInstance)
            critic_copy.reset_parameters()
            self.critics.append(critic_copy)

    def __getattr__(self, attr):
        if attr == 'critics':
            return self._modules['critics']
        return getattr(self.critics[0], attr)

    def forward(self, *args, **kwargs):
        qvalue_list = [critic(*args, **kwargs) for critic in self.critics]
        return qvalue_list

    def state_dict(self, *args, **kwargs):
        state_dict = {}
        for i, critic in enumerate(self.critics):
            state_dict[f'critic_{i}'] = critic.state_dict(*args, **kwargs)
        return state_dict

    def load_state_dict(self, state_dict):
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(state_dict[f'critic_{i}'])


from .ddpg import __ModuleCls__

builder.add_to_registry(__ModuleCls__, [EnsembledCritic])


class DistributionalMixin:  # from SoRB
    def get_qvalues(self, state, actions=None, aggregate='default'):
        if aggregate == 'default':
            aggregate = self.critic_aggregate

        qvalues = super().get_qvalues(state, actions=actions)

        if self.critic_ensemble_size == 1:
            qvalues_list = [qvalues]
        else:
            qvalues_list = qvalues

        distributional = self.critic.is_distributional
        num_bins = self.critic.num_bins
        clip_q = self.critic.clip_q

        expected_qvalues_list = []
        if distributional:
            for qvalues in qvalues_list:
                q_probs = F.softmax(qvalues, dim=1)
                batch_size = q_probs.shape[0]
                # NOTE: We want to compute the value of each bin, which is the
                # negative distance. Without properly negating this, the actor is
                # optimized to take the *worst* actions.
                neg_bin_range = -torch.arange(0, num_bins, dtype=torch.float, device=qvalues.device)
                # neg_bin_range = -torch.arange(1, num_bins + 1, dtype=torch.float)
                tiled_bin_range = neg_bin_range.unsqueeze(0).repeat(batch_size, 1)
                assert q_probs.shape == tiled_bin_range.shape
                # Take the inner product between these two tensors
                expected_qvalues = torch.sum(q_probs * tiled_bin_range, dim=1, keepdim=True)
                expected_qvalues_list.append(expected_qvalues)
        else:
            expected_qvalues_list = qvalues_list

        expected_qvalues = torch.stack(expected_qvalues_list)
        if aggregate is not None:
            if aggregate == 'mean':
                expected_qvalues = torch.mean(expected_qvalues, dim=0)
            elif aggregate == 'min':
                expected_qvalues, _ = torch.min(expected_qvalues, dim=0)
            elif aggregate == 'max':
                expected_qvalues, _ = torch.max(expected_qvalues, dim=0)
            else:
                raise ValueError

        if not distributional and clip_q:
            # Clip the q values if not using distributional RL. If using
            # distributional RL, the q values are implicitly clipped.
            min_qvalue = -1.0 * self.max_q
            max_qvalue = 0.0
            expected_qvalues = torch.clamp(expected_qvalues, min_qvalue, max_qvalue)

        return expected_qvalues

    def critic_loss(self, current_q, target_q, reward, done, discount):
        if not isinstance(current_q, list):
            current_q_list = [current_q]
            target_q_list = [target_q]
        else:
            current_q_list = current_q
            target_q_list = target_q

        distributional = self.critic.is_distributional
        num_bins = self.critic.num_bins
        clip_q = self.critic.clip_q

        if not distributional and clip_q:
            # Clip the q values if not using distributional RL. If using
            # distributional RL, the q values are implicitly clipped.
            min_qvalue = -1.0 * self.max_q
            max_qvalue = 0.0
            current_q_list = [torch.clamp(x, min_qvalue, max_qvalue) for x in current_q_list]
            target_q_list = [torch.clamp(x, min_qvalue, max_qvalue) for x in target_q_list]

        critic_loss_list = []
        for current_q, target_q in zip(current_q_list, target_q_list):  # iterate over ensemble
            if distributional:
                if discount != 1:
                    raise RuntimeError(f'discount must be 1, got {discount}')

                # Compute distributional td targets
                target_q_probs = F.softmax(target_q, dim=1)
                batch_size = target_q_probs.shape[0]
                one_hot = torch.zeros((batch_size, num_bins), device=target_q.device)
                one_hot[:, 0] = 1
                # Calculate the shifted probabilities
                # Fist column: Since episode didn't terminate, probability that the
                # distance is 1 equals 0.
                col_1 = torch.zeros((batch_size, 1), device=target_q.device)
                # Middle columns: Simply the shifted probabilities.
                col_middle = target_q_probs[:, :-2]
                # Last column: Probability of taking at least n steps is sum of
                # last two columns in unshifted predictions:
                col_last = torch.sum(target_q_probs[:, -2:], dim=1, keepdim=True)
                shifted_target_q_probs = torch.cat([col_1, col_middle, col_last], dim=1)
                assert one_hot.shape == shifted_target_q_probs.shape
                # use .byte() instead of .bool() for torch < 1.3
                td_targets = torch.where(done.byte(), one_hot, shifted_target_q_probs).detach()

                critic_loss = torch.mean(
                    -torch.sum(td_targets * torch.log_softmax(current_q, dim=1), dim=1)
                )  # https://github.com/tensorflow/tensorflow/issues/21271
            else:
                critic_loss = super().critic_loss(current_q, target_q, reward, done, discount)
            critic_loss_list.append(critic_loss)
        critic_loss = torch.mean(torch.stack(critic_loss_list))
        return critic_loss


class UVFDDPG(DistributionalMixin, DDPG):
    def __init__(
        self,
        mdp_space,
        critic_params=dict(cls='Critic'),
        use_distributional_critic=False,
        max_q=None,
        critic_ensemble_size=1,
        critic_aggregate='mean',
        **kwargs,
    ):
        self.use_distributional_critic = use_distributional_critic
        self.max_q = max_q
        self.critic_ensemble_size = critic_ensemble_size
        self.critic_aggregate = critic_aggregate

        if use_distributional_critic:
            critic_params.update(output_dim=max_q + 1)
        if max_q is not None:
            critic_params.update(clip_q=True)

        super().__init__(mdp_space, critic_params=critic_params, **kwargs)

        if self.use_distributional_critic:
            if not self.discount == 1:
                raise RuntimeError('Must set discount=1 if use_distributional_critic.')

        if self.critic_ensemble_size > 1:
            self.critic = EnsembledCritic(
                self.critic, critic_ensemble_size=self.critic_ensemble_size
            )
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_target.load_state_dict(self.critic.state_dict())

            # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
            for i in range(1, len(self.critic.critics)):  # first copy already added
                critic_copy = self.critic.critics[i]
                self.critic_optimizer.add_param_group({'params': critic_copy.parameters()})
                # https://stackoverflow.com/questions/51756913/in-pytorch-how-do-you-use-add-param-group-with-a-optimizer
