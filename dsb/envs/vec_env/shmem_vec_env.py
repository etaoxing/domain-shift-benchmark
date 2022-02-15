import multiprocessing
from multiprocessing import shared_memory
from collections import OrderedDict
from typing import Sequence
import time
import collections
import atexit

import gym
import numpy as np

from .base_vec_env import VecEnv, CloudpickleWrapper
from .util import obs_space_info


def _worker(remote, parent_remote, lock, env_fn_wrapper, obs_bufs, obs_shapes, obs_dtypes, keys):
    def _write_obs(maybe_dict_obs):
        flatdict = obs_to_dict(maybe_dict_obs)
        for k in keys:
            dst = shared_memory.SharedMemory(name=obs_bufs[k])
            dst_np = np.ndarray(obs_shapes[k], dtype=obs_dtypes[k], buffer=dst.buf)
            np.copyto(dst_np, flatdict[k])

            del dst_np  # Unnecessary; merely emphasizing the array is no longer used
            dst.close()

    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    # save final observation where user can get it, then reset
                    info['terminal_observation'] = observation
                    observation = env.reset()
                lock.acquire()
                o = _write_obs(observation)
                lock.release()
                remote.send((o, reward, done, info))
            elif cmd == 'seed':
                remote.send(env.seed(data))
            elif cmd == 'reset':
                observation = env.reset(**data)
                lock.acquire()
                o = _write_obs(observation)
                lock.release()
                remote.send(o)
            elif cmd == 'render':
                remote.send(env.render(data))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                try:
                    o = getattr(env, data)
                except AttributeError:
                    o = None
                remote.send(o)
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


# from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/shmem_vec_env.py
class ShmemVecEnv(VecEnv):
    """
    Optimized version of SubprocVecEnv that uses shared variables to communicate observations.
    """

    def __init__(self, env_fns, start_method='forkserver'):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        self.stats = collections.defaultdict(int)

        # Creating dummy env object to get spaces
        dummy = env_fns[0]()
        observation_space, action_space = dummy.observation_space, dummy.action_space
        dummy.close()
        del dummy
        VecEnv.__init__(self, n_envs, observation_space, action_space)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(observation_space)

        self.obs_bufs = [{} for _ in range(n_envs)]
        for k in self.obs_keys:
            nbytes = int(np.prod(self.obs_shapes[k]) * np.dtype(self.obs_dtypes[k].type).itemsize)
            for i in range(n_envs):
                self.obs_bufs[i][k] = shared_memory.SharedMemory(create=True, size=nbytes)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        self.locks = []
        for work_remote, remote, env_fn, obs_buf in zip(
            self.work_remotes, self.remotes, env_fns, self.obs_bufs
        ):
            lock = ctx.Lock()
            obs_buf_shm_name = {k: v.name for k, v in obs_buf.items()}
            args = (
                work_remote,
                remote,
                lock,
                CloudpickleWrapper(env_fn),
                obs_buf_shm_name,
                self.obs_shapes,
                self.obs_dtypes,
                self.obs_keys,
            )
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(
                target=_worker, args=args, daemon=True
            )  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            self.locks.append(lock)
            work_remote.close()

        atexit.register(self.close)

    def _decode_obses(self, obs):
        for lock in self.locks:
            lock.acquire()

        result = {}
        for k in self.obs_keys:
            bufs = [b[k] for b in self.obs_bufs]
            o = [
                np.ndarray(self.obs_shapes[k], dtype=self.obs_dtypes[k], buffer=b.buf) for b in bufs
            ]
            result[k] = np.array(o)

        for lock in self.locks:
            lock.release()

        return dict_to_obs(result)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = []
        for i, remote in enumerate(self.remotes):
            start = time.perf_counter()
            r = remote.recv()
            end = time.perf_counter()
            self.stats[f'env{i}_step'] += end - start
            results.append(r)
        # results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return self._decode_obses(obs), np.vstack(rews), np.vstack(dones), infos

    def seed(self, seed=None):
        seeds = self._get_seeds(seed, len(self.remotes))
        for idx, remote in enumerate(self.remotes):
            remote.send(('seed', seeds[idx]))
        return [remote.recv() for remote in self.remotes]

    def reset(self, **reset_kwargs):
        for remote in self.remotes:
            remote.send(('reset', reset_kwargs))
        obs = [remote.recv() for remote in self.remotes]
        return self._decode_obses(obs)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

        for obs_buf in self.obs_bufs:
            for k, v in obs_buf.items():
                v.close()
                v.unlink()

    def get_images(self) -> Sequence[np.ndarray]:
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def dict_to_obs(obs_dict):
    """
    Convert an observation dict into a raw array if the
    original observation space was not a Dict space.
    """
    if set(obs_dict.keys()) == {None}:
        return obs_dict[None]
    return obs_dict


def obs_to_dict(obs):
    """
    Convert an observation into a dict.
    """
    if isinstance(obs, dict):
        return obs
    return {None: obs}
