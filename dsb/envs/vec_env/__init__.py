# from https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/common/vec_env
from .base_vec_env import VecEnv
from .dummy_vec_env import DummyVecEnv

try:
    from .shmem_vec_env import ShmemVecEnv
except ImportError as e:
    print(f'{e}, ShmemVecEnv unavailable, use SubprocVecEnv instead')
from .subproc_vec_env import SubprocVecEnv
