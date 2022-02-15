from .buffer_wrapper import BufferWrapper


class SampleBufferWrapper(BufferWrapper):
    # Inherit from this class if want to override buffer.sample so calling
    # the new wrapper's sample_ptr and/or get_batch.
    # This is used to help keep things process safe

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override this if want to change behavior
    def sample_ptrs(self, batch_it, batch_size):
        return self.buffer.sample_ptrs(batch_it, batch_size)

    # override this if want to change behavior
    def get_batch(self, batch_it, ptrs):
        batch = self.buffer.get_batch(batch_it, ptrs)
        return ptrs, batch

    # probably don't need to override this
    def sample(self, batch_it, batch_size):
        ptrs = self.sample_ptrs(batch_it, batch_size)
        return self.get_batch(batch_it, ptrs)
