from dsb.dependencies import *

from .base_policy import BasePolicy


class RPLPolicy(BasePolicy):
    # use to keep high_action <-> subgoal for low policy the same, so high policy only
    # actions every decision_interval steps

    def __init__(self, agent, decision_interval=30, **kwargs):
        super().__init__(agent, **kwargs)
        # interval for high-level policy to select a new high action (goal for low-level policy)
        self.decision_interval = decision_interval

        # key is env_idx
        self.elapsed_since_high = {}
        self.cached_high_action = {}

    def select_action(self, state, deterministic=False, **kwargs):
        num_envs = len(state['_time_step'])
        high_action_mask = np.zeros(num_envs, dtype=np.bool)
        high_action = state['desired_goal'].copy()

        for env_idx in range(num_envs):
            episode_start = state['_time_step'][env_idx] == 0
            if episode_start:
                high_action_mask[env_idx] = True
                self.elapsed_since_high[env_idx] = 0
            else:
                self.elapsed_since_high[env_idx] += 1
                if self.elapsed_since_high[env_idx] >= self.decision_interval:
                    high_action_mask[env_idx] = True
                    self.elapsed_since_high[env_idx] = 0
                else:
                    # high_action[env_idx, ...] = self.cached_high_action[env_idx].copy()
                    # TODO: copy into state_embedding instead
                    pass

        state['high_action'] = high_action
        state, action = self.agent.select_action(
            state, high_action_mask=high_action_mask, deterministic=deterministic, **kwargs
        )

        for env_idx in range(num_envs):
            if high_action_mask[env_idx]:
                self.cached_high_action[env_idx] = state['high_action'][env_idx, ...].copy()

        return state, action
