from dsb.dependencies import *

from .base_policy import BasePolicy


class GaussianNoisePolicy(BasePolicy):
    def __init__(self, agent, noise_scale=1.0, **kwargs):
        super().__init__(agent, **kwargs)
        self.noise_scale = noise_scale

    def select_action(self, state, deterministic=False, **kwargs):
        state, action = self.agent.select_action(state, deterministic=deterministic, **kwargs)
        if not deterministic:
            noise = np.rng.normal(
                0,
                self.agent.max_action * self.noise_scale,
                size=action.shape,
            )
            action += noise
            action = action.clip(-self.agent.max_action, self.agent.max_action)
        return state, action
