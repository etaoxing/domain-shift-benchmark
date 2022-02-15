from dsb.dependencies import *

# https://www.tensorflow.org/agents/tutorials/2_environments_tutorial#using_standard_environments
# step_type: first_step=0, step=1, last_step=2
class TimeStep:
    def __init__(self):
        pass


# https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py
class TimeLimitWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        max_episode_steps=None,
        reset_on='either',
        auto_reset=False,
        disable_rewards=False,
    ):
        assert reset_on in ['timeout', 'success', 'either']
        super().__init__(env)
        self.duration = max_episode_steps
        self._elapsed_steps = None
        self.reset_on = reset_on
        self.auto_reset = auto_reset
        self.disable_rewards = disable_rewards

        # environment steps vs. interaction steps, see https://arxiv.org/pdf/2004.04136.pd
        self.observation_space.spaces.update(
            {
                '_time_step': gym.spaces.Box(
                    low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32
                )
            }
        )

    @property
    def duration(self):
        return self._max_episode_steps

    @duration.setter
    def duration(self, max_episode_steps):
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = self.env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, success, info = self.env.step(action)

        if self.disable_rewards:
            success = False

        self._elapsed_steps += 1
        timeout = self._elapsed_steps >= self._max_episode_steps
        observation['_time_step'] = self._elapsed_steps
        info['success'] = success
        info['timeout'] = timeout

        done = (
            (success and self.reset_on == 'success')
            or (timeout and self.reset_on == 'timeout')
            or ((success or timeout) and self.reset_on == 'either')
        )
        if done:
            if self.auto_reset:  # do not use w/ vec_env, since it handles auto resetting
                info['terminal_observation'] = observation  # same as vec_env
                observation = self.reset()
            else:
                self._elapsed_steps = None
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        observation = self.env.reset(**kwargs)
        observation['_time_step'] = self._elapsed_steps
        return observation


class SparseRewardWrapper(gym.RewardWrapper):  # use before TimeLimit wrapper
    def __init__(self, env, zero_reward_on_success):
        super().__init__(env)
        self.zero_reward_on_success = zero_reward_on_success

    def step(self, action):
        obs, reward, success, info = self.env.step(action)

        reward = 0.0 if (success and self.zero_reward_on_success) else reward
        return obs, reward, success, info

    def reached_goal(self, state):
        reward, success = self.env.reached_goal(state)

        if self.zero_reward_on_success:
            if isinstance(reward, np.ndarray):
                reward[success.astype(np.bool)] = 0.0
            elif success:
                reward = 0.0
        return reward, success
