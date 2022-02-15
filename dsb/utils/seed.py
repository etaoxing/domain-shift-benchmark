from dsb.dependencies import *

from numpy.random import SeedSequence, Generator, SFC64


def set_global_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    # torch.cuda.manual_seed_all(seed) # cuda.manual_seed only seeds the current gpu
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    set_worker_seed(seed)


# PCG64 issues, see https://github.com/numpy/numpy/issues/16313
# https://prng.di.unimi.it
# could also update https://numpy.org/doc/stable/reference/random/upgrading-pcg64.html
def make_rng(seed):
    return Generator(SFC64(seed))


# https://pytorch.org/docs/master/notes/randomness.html#dataloader
# https://github.com/pytorch/pytorch/issues/5059
def set_worker_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    np.rng = make_rng(seed)


def set_env_seed(env, seed):
    env.seed(seed)
    # https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/
    env.action_space.seed(seed)
    env.base_seed = seed
