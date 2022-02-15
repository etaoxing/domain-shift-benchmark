from dsb.dependencies import *


class BaseStorage:
    def __init__(
        self,
        obs_space,
        action_space,
        max_size,
        tmp_dir=None,
        batch_dtype=None,
    ):
        self.obs_space = obs_space
        self.action_space = action_space
        self.max_size = max_size
        self.tmp_dir = tmp_dir

        # cast all arrays sampled on batch to the specified dtype
        # if None, then keep original dtype saved
        # only use this w/ dataloader and pin_memory=True
        self.batch_dtype = np.dtype(batch_dtype) if batch_dtype is not None else None
        # TODO: don't convert for state, next_state if starts with '_'

        self.cache = {}

    def reset_stats(self):
        pass

    def get_stats(self):
        return {}

    def get_cached(self, key):
        if key in self.cache.keys():
            return True, self.cache[key]
        else:
            return False, None

    def save_in_cache(self, key, value):
        self.cache[key] = value

    def clear_cache(self):
        self.cache.clear()
