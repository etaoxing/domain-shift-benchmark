import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import VisionTransformer


def _get_out_shape_cuda(in_shape, layers):
    x = torch.randn(*in_shape).cuda().unsqueeze(0)
    return layers(x).squeeze(0).shape

def _get_out_shape(in_shape, layers):
    x = torch.randn(*in_shape).unsqueeze(0)
    return layers(x).squeeze(0).shape


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)


class SharedTransformer(nn.Module):
    def __init__(self, obs_shape, patch_size=8, embed_dim=128, depth=4, num_heads=8, mlp_ratio=1., qvk_bias=False):
        super().__init__()
        assert len(obs_shape) == 3
        # self.frame_stack = obs_shape[0]//3
        self.in_chans = obs_shape[0]
        self.img_size = obs_shape[-1]
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qvk_bias = qvk_bias

        # self.preprocess = nn.Sequential(CenterCrop(size=self.img_size), NormalizeImg())
        self.transformer = VisionTransformer(
            img_size=self.img_size,
            patch_size=patch_size,
            in_chans=self.in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qvk_bias,
        )
        self.out_shape = _get_out_shape(obs_shape, nn.Sequential(self.transformer))

    def forward(self, x):
        # x = self.preprocess(x)
        return self.transformer(x)


class HeadCNN(nn.Module):
    def __init__(self, in_shape, num_layers=0, num_filters=32):
        super().__init__()
        self.layers = []
        for _ in range(0, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers.append(Flatten())
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(in_shape, self.layers)
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x)
