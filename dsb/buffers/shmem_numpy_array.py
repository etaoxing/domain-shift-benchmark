import numpy as np
import multiprocessing

try:
    # raise ImportError

    from multiprocessing import shared_memory

    # from multiprocessing.managers import SharedMemoryManager
    # import atexit
    # smm = SharedMemoryManager()
    # smm.start()
    # atexit.register(smm.shutdown) # care, https://stackoverflow.com/questions/16333054/what-are-the-implications-of-registering-an-instance-method-with-atexit-in-pytho

    class ShmemNumpyArray(object):
        def __init__(self, shape, dtype=np.float32, lock=None):
            self._shape = shape
            self._dtype = dtype

            nbytes = int(np.prod(self._shape) * np.dtype(dtype).itemsize)
            self._shmem = shared_memory.SharedMemory(create=True, size=nbytes)
            # self._shmem = smm.SharedMemory(nbytes)

            if lock is None:
                self._lock = multiprocessing.Lock()
            else:
                self._lock = lock
            self._numpy = self.numpy()

        def __getattr__(self, attr):
            if attr == '_numpy':
                raise AttributeError
            return getattr(self._numpy, attr)

        def __len__(self):
            return len(self._numpy)

        def __getstate__(self):
            state = self.__dict__.copy()
            state['_shmem_name'] = state['_shmem'].name
            del state['_shmem']
            del state['_numpy']
            return state

        def __setstate__(self, state):
            _shmem_name = state.pop('_shmem_name')
            self.__dict__.update(state)
            self._shmem = shared_memory.SharedMemory(name=_shmem_name)
            self._numpy = self.numpy()

        def __getitem__(self, idx):
            self._lock.acquire()
            y = self._numpy[idx]
            self._lock.release()
            return y

        def __setitem__(self, key, value):
            self._lock.acquire()
            self._numpy[key] = value
            self._lock.release()

        def numpy(self):
            return np.ndarray(self._shape, dtype=self._dtype, buffer=self._shmem.buf)

        def __del__(self):
            if self._shmem is not None:  # check if free already called
                self._shmem.close()

        def free(self):
            self._shmem.close()
            self._shmem.unlink()
            self._shmem = None
            self._lock = None


except ImportError as e:

    from multiprocessing import current_process

    if current_process().name == 'MainProcess':
        print(f'{e}, falling back on CtypesShmemNumpyArray')

    import ctypes

    _TO_CT = {
        np.float32: ctypes.c_float,
        np.int32: ctypes.c_int32,
        np.int8: ctypes.c_int8,
        np.uint8: ctypes.c_char,
        np.bool: ctypes.c_bool,
    }
    _TO_CT = {np.dtype(k): v for k, v in _TO_CT.items()}

    class CtypesShmemNumpyArray(object):
        def __init__(self, shape, dtype=np.float32, lock=None):
            self._shape = shape
            self._dtype = np.dtype(dtype)
            self._storage = multiprocessing.Array(
                _TO_CT[self._dtype],
                int(np.prod(self._shape)),
                lock=True if lock is None else lock,
            )  # this should zero out memory, also has built in lock
            self._numpy = self.numpy()

        def __getattr__(self, attr):
            if attr == '_numpy':
                raise AttributeError
            return getattr(self._numpy, attr)

        def __len__(self):
            return len(self._numpy)

        def __getstate__(self):
            state = self.__dict__.copy()
            del state['_numpy']
            return state

        def __setstate__(self, state):
            self.__dict__.update(state)
            self._numpy = self.numpy()

        def __getitem__(self, idx):
            return self._numpy[idx]

        def __setitem__(self, key, value):
            self._numpy[key] = value

        def numpy(self):
            return np.frombuffer(self._storage.get_obj(), dtype=self._dtype).reshape(self._shape)

        def free(self):
            self._storage = None

    ShmemNumpyArray = CtypesShmemNumpyArray
