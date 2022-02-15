from dsb.dependencies import *

from .base_policy import BasePolicy


class PlaybackDemoPolicy(BasePolicy):
    # playback the actions given the context demo
    # Use with ContextDemoPolicy
    def __init__(self, agent, buffer=None, **kwargs):
        super().__init__(agent, **kwargs)

    def select_action(self, state, deterministic=False, **kwargs):
        state, _ = self.agent.select_action(state, deterministic=deterministic, **kwargs)
        # overwriting action

        num_envs = len(state['_time_step'])
        action = []
        for env_idx in range(num_envs):
            t = state['_time_step'][env_idx]

            demo = self.context_demo_cached[env_idx]
            N = len(demo['action'])
            a = demo['action'][np.clip(t, 0, N - 1), ...]
            action.append(a)

        action = np.concatenate(action)
        return state, action
