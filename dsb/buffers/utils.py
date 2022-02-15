from dsb.dependencies import *
from dsb.utils import set_worker_seed


def worker_init_fn(i):
    s = int(torch.initial_seed()) % (2 ** 32 - 1)
    set_worker_seed(s)


def get_timestep_of(ptrs, episode_ptrs):
    # ptrs is an array of transition ptrs belonging to the same episode_start_ptr
    # episode_ptrs is an array of ptrs belonging to the same episode
    y = np.array(ptrs)
    x = np.array(episode_ptrs)
    # find the index t of every element of y in x
    t = np.where(y.reshape(-1, 1) == x)[1]
    assert len(t) == len(y)  # check if found every element
    return t


# https://github.com/SudeepDasari/one_shot_transformers/blob/ecd43b0c182451b67219fdbc7d6a3cd912395f17/hem/datasets/agent_dataset.py#L91
def sample_context_indices(N, T_context, sample_sides=False):
    per_bracket = max(N / T_context, 1)
    selected_t = []
    for i in range(T_context):
        if sample_sides and i == T_context - 1:
            t = N - 1
        elif sample_sides and i == 0:
            t = 0
        else:
            t = np.rng.integers(int(i * per_bracket), int((i + 1) * per_bracket))
            t = max(t, N - 1)
        selected_t.append(t)
    return np.array(selected_t)


# samples from [indices + low, indices + high] bounded between [0, N)
def sample_pos_pair_indices(indices, N, low, high=None):
    assert np.all(indices) < N
    assert np.all(indices) >= 0

    if high is None:
        high = N - 1  # could also use np.inf, but would need to cast to int
    else:
        assert high > 0
    assert low > 0
    assert low < N

    # removing this check b/c if given terminal_ptr of episode, then just pair it with itself
    # assert np.all((indices + low) < N) # desired interval should intersection w/ [0, N)

    l = np.minimum(indices + low, N - 1)
    r = np.minimum(indices + high, N - 1)

    # it's better to take the minimum on the bounds rather than the output, otherwise
    # would skew the distribution toward the tail if high >> N
    # ie. pos_pair = np.minimum(indices + np.rng.integers(low, high + 1, size=len(indices)), N - 1)
    # is less desirable

    pos_pair = np.rng.integers(l, r + 1)
    assert len(pos_pair) == len(indices)
    return pos_pair


# samples from [indices - high, indices - low] U [indices + low, indices + high] bounded between [0, N)
# if high == None, then samples from [0, indices - low] U [indices + low, N)
def sample_neg_pair_indices(indices, N, low, high=None):
    assert np.all(indices) < N
    assert np.all(indices) >= 0

    if high is None:
        high = N - 1  # could also use np.inf, but would need to cast to int
    else:
        assert high > 0
    assert low > 0
    assert low < N

    assert np.all(
        (indices - low >= 0) | (indices + low < N)
    )  # desired interval should intersection w/ [0, N)

    left_interval_l = np.maximum(0, indices - high)
    left_interval_r = np.maximum(0, indices - low)

    right_interval_l = np.minimum(indices + low, N - 1)
    right_interval_r = np.minimum(indices + high, N - 1)

    mask = (
        np.rng.uniform(size=len(indices)) < 0.5
    )  # if True, then use left interval, otherwise sample from right interval

    mask[indices - low < 0] = False
    mask[indices + low >= N] = True

    neg_pair = np.zeros(len(indices), dtype=np.int)

    npl = np.rng.integers(left_interval_l, left_interval_r + 1)
    npr = np.rng.integers(right_interval_l, right_interval_r + 1)

    assert len(npl) == len(indices)
    assert len(npr) == len(indices)

    neg_pair[mask] = npl[mask]
    neg_pair[~mask] = npr[~mask]
    return neg_pair

    # ported code below only works on integer inputs
    # from https://github.com/nsavinov/SPTM/blob/45592e5f86b3c509665e4d72c756633489a7124c/src/train/train_edge_predictor.py#L23

    # current_second_before = None
    # current_second_after = None
    # index_before_max = current_first - low
    # index_after_min = current_first + low
    # # NOTE: random.randint acts on [a, b] closed interval
    # # while np.random.randint uses [a, b)
    # if index_before_max >= 0:
    #     l = 0 if high is None else np.maximum(0, current_first - high)
    #     current_second_before = np.random.randint(l, index_before_max + 1)
    # if index_after_min < N: # MAX_CONTINUOUS_PLAY == length of demo trajectory
    #     h = N if high is None else np.minimum(current_first + high + 1, N)
    #     current_second_after = np.random.randint(index_after_min, h)
    # if current_second_before is None:
    #     current_second = current_second_after
    # elif current_second_after is None:
    #     current_second = current_second_before
    # else:
    #     if np.random.random() < 0.5:
    #         current_second = current_second_before
    #     else:
    #         current_second = current_second_after


def test_sample_neg_pair_indices(test_func, output_range, num_per_test=1000):
    remaining_output_range = copy.deepcopy(output_range)
    for _ in range(num_per_test):
        o_all = test_func()
        for o in o_all:
            if o in output_range:
                remaining_output_range.discard(o)
            else:
                raise Exception(f'got output {o} but expected output range is {output_range}')
        # if len(remaining_output_range) == 0:
        #     break
    if len(remaining_output_range) > 0:
        raise Exception(f'did not get output {remaining_output_range}')


# if __debug__:
# if __name__ == '__main__':
if False:
    tests = [
        [[[1], 10, 2, 2], [3]],
        [[[1], 10, 1, 1], [0, 2]],
        [[[1], 10, 1, 5], [0, 2, 3, 4, 5, 6]],
        [[[1], 10, 2, 5], [3, 4, 5, 6]],
        [[[15, 16], 20, 1, 1], [14, 15, 16, 17]],
        [[[2], 6, 2], [0, 4, 5]],
        [[[2], 6, 2, 10], [0, 4, 5]],
        [[[0], 6, 2], [2, 3, 4, 5]],
        [[[4], 6, 2], [0, 1, 2]],
        [[[4], 5, 2, 2], [2]],
        [[[2], 5, 2, 100], [0, 4]],
        # [[[1], 3, 2, 2], []], # this should fail
    ]
    for i, (x, y) in enumerate(tests):
        x[0] = np.array(x[0])
        y = set(y)
        try:
            test_sample_neg_pair_indices(functools.partial(sample_neg_pair_indices, *x), y)
        except Exception as e:
            print(f'failed test {i}')
            raise e
    # NOTE: run w/ https://docs.python.org/3.3/using/cmdline.html#cmdoption-OO to disable asserts
