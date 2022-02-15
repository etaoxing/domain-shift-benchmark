from dsb.dependencies import *

from .buffer_wrapper import BufferWrapper
from ..storage.base_storage import BaseStorage


class DatasetBufferWrapper(BufferWrapper):  # holds datasets for policy to access
    def __init__(
        self,
        buffer,
        datasets=None,
        augment_batch_with_demos=False,
        demo_buffer_wrappers=[],
        demo_batch_size=128,
        **kwargs
    ):
        super().__init__(buffer, **kwargs)
        self.datasets = datasets
        self.augment_batch_with_demos = augment_batch_with_demos
        self.demo_batch_size = demo_batch_size

        if self.augment_batch_with_demos:
            from dsb.builder import build_buffer

            self.demo_buffer = build_buffer(
                dict(
                    cls='DemonstrationsBuffer',
                    dataloader_params=dict(shuffle=True, batch_size=demo_batch_size, num_workers=0),
                    buffer_wrappers=demo_buffer_wrappers,
                ),
                BaseStorage(self.obs_space, self.action_space, None),
                datasets=datasets,
            )
            self.demo_buffer.init_dataloader('train_dataset')

    def sample_from_demos(self):
        return self.demo_buffer.sample(0, self.demo_batch_size)

    def clear_cache(self):
        if hasattr(self.buffer, 'clear_cache'):
            if self.augment_batch_with_demos:
                self.demo_buffer.clear_cache()
            self.buffer.clear_cache()
