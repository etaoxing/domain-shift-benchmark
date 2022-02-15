from dsb.dependencies import *

from dsb.utils import torchify, untorchify, compute_output_shape
import dsb.builder as builder
from dsb.builder import module_reset_parameters


class BasicCNN(nn.Module):
    def __init__(
        self,
        obs_space,
        encoder_network_params=[],
        output_activation_params=None,
        embedding_dim=512,
        detach_embedding=False,
    ):
        super().__init__()
        assert len(obs_space.shape) == 3  # check if image space
        self.obs_space = obs_space
        self.embedding_dim = embedding_dim
        self.detach_embedding = detach_embedding

        in_channels = self.obs_space.shape[0]
        body = builder.build_network_modules(encoder_network_params, in_channels=in_channels)
        self.body = nn.Sequential(*body)

        self.conv_output_shape, n_flatten = compute_output_shape(self.obs_space.sample(), self.body)

        self.fc = nn.Linear(n_flatten, self.embedding_dim)
        if output_activation_params is not None:
            self.output_activation = builder.build_network_modules([output_activation_params])[0]
        else:
            self.output_activation = None

        self.reset_parameters()

    def reset_parameters(self):
        self.body.apply(module_reset_parameters)
        self.fc.reset_parameters()

    def forward(self, x, with_conv_output=False, detach_embedding=None):
        conv_output = self.body(x)
        h = conv_output.flatten(start_dim=1)
        z = self.fc(h)
        if self.output_activation is not None:
            z = self.output_activation(z)

        detach_embedding = self.detach_embedding if detach_embedding is None else detach_embedding
        if detach_embedding:
            z = z.detach()

        if with_conv_output:
            return z, conv_output
        else:
            return z

    @property
    def opt_info_keys(self):
        return []
