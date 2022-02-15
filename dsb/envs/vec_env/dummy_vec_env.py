from collections import OrderedDict
from copy import deepcopy
from typing import Sequence

import gym
import numpy as np

from .base_vec_env import VecEnv
from .util import obs_space_info


def copy_obs_dict(obs):
    """
    Deep-copy a dict of numpy arrays.

    :param obs: (OrderedDict<ndarray>): a dict of numpy arrays.
    :return (OrderedDict<ndarray>) a dict of copied numpy arrays.
    """
    assert isinstance(obs, OrderedDict), f"unexpected type for observations '{type(obs)}'"
    return OrderedDict([(k, np.copy(v)) for k, v in obs.items()])


def dict_to_obs(space, obs_dict):
    """
    Convert an internal representation raw_obs into the appropriate type
    specified by space.

    :param space: (gym.spaces.Space) an observation space.
    :param obs_dict: (OrderedDict<ndarray>) a dict of numpy arrays.
    :return (ndarray, tuple<ndarray> or dict<ndarray>): returns an observation
            of the same type as space. If space is Dict, function is identity;
            if space is Tuple, converts dict to Tuple; otherwise, space is
            unstructured and returns the value raw_obs[None].
    """
    if isinstance(space, gym.spaces.Dict):
        return obs_dict
    elif isinstance(space, gym.spaces.Tuple):
        assert len(obs_dict) == len(
            space.spaces
        ), "size of observation does not match size of observation space"
        return tuple((obs_dict[i] for i in range(len(space.spaces))))
    else:
        assert set(obs_dict.keys()) == {
            None
        }, "multiple observation keys for unstructured observation space"
        return obs_dict[None]


class DummyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: ([Gym Environment]) the list of environments to vectorize
    """

    def __init__(self, env_fns, start_method=None):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        assert env is not None
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        # o = []
        # for k in self.keys:
        #     if shapes[k] == (1,):
        #         b = (k, np.zeros((self.num_envs), dtype=dtypes[k]))
        #     else:
        #         b = (k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]))
        #     o.append(b)
        # self.buf_obs = OrderedDict(o)
        # self.buf_dones = np.zeros((self.num_envs), dtype=np.bool)
        # self.buf_rews = np.zeros((self.num_envs), dtype=np.float32)

        self.buf_obs = OrderedDict(
            [(k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k])) for k in self.keys]
        )
        self.buf_dones = np.zeros(
            (self.num_envs, 1), dtype=np.bool
        )  # shape (num_envs, 1) to match subproc_vec_env
        self.buf_rews = np.zeros((self.num_envs, 1), dtype=np.float32)

        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = env.metadata

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for env_idx in range(self.num_envs):
            (
                obs,
                self.buf_rews[env_idx],
                self.buf_dones[env_idx],
                self.buf_infos[env_idx],
            ) = self.envs[env_idx].step(self.actions[env_idx])
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]['terminal_observation'] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def seed(self, seed=None):
        seeds = self._get_seeds(seed, len(self.envs))
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seeds[idx]))
        return seeds

    def reset(self, **reset_kwargs):
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset(**reset_kwargs)
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def close(self):
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[np.ndarray]:
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode: str = 'human'):
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.
        Otherwise (if ``self.num_envs == 1``), we pass the render call directly to the
        underlying environment.

        Therefore, some arguments such as ``mode`` will have values that are valid
        only when ``num_envs == 1``.

        :param mode: The rendering type.
        """
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def _save_obs(self, env_idx, obs):
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def _obs_from_buf(self):
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def _get_target_envs(self, indices):
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
