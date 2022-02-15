from dsb.dependencies import *
from dsb.utils import torchify, untorchify


class RewardShaper:
    def __init__(self, agent, sub_type='embedding_dist', reward_scaling=None):
        self.agent = agent
        self.sub_type = sub_type
        self.reward_scaling = reward_scaling

    def __call__(self, state, reward, done, state_embedding=None):  # batch_size == num_envs
        # not inplace so should only contain embeddings
        state_embedding = self.agent.compute_embedding(
            torchify(state), inplace=False, state_embedding=state_embedding
        )

        if self.sub_type == 'embedding_dist':  # RIG, -1 * ||z_s - z_g||_2
            sub_reward = -torch.norm(
                state_embedding['achieved_goal'] - state_embedding['desired_goal'], dim=1, p=2
            ).view(-1, 1)
            if self.reward_scaling is not None:
                sub_reward *= self.reward_scaling
        elif self.sub_type == 'learned_dist':
            pass
        else:
            raise ValueError
        sub_reward = untorchify(sub_reward)

        mask = ~done  # if done, then keep original reward
        reward[mask] = sub_reward[mask]
        return state_embedding, reward
