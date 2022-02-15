import torch.nn as nn

# https://github.com/vitchyr/rlkit/blob/v0.1.2/rlkit/torch/vae/conv_vae.py#L13
# used for Skewfit and RIG papers
imsize48_encoder = [
    dict(cls='Conv2d', out_channels=16, kernel_size=5, stride=3),
    dict(cls='ReLU', inplace=True),
    dict(cls='Conv2d', out_channels=32, kernel_size=3, stride=2),
    dict(cls='ReLU', inplace=True),
    dict(cls='Conv2d', out_channels=64, kernel_size=3, stride=2),
    dict(cls='ReLU', inplace=True),
]
imsize48_decoder = [
    dict(cls='ConvTranspose2d', in_channels=64, out_channels=32, kernel_size=3, stride=2),
    dict(cls='ReLU', inplace=True),
    dict(cls='ConvTranspose2d', out_channels=16, kernel_size=3, stride=2),
    dict(cls='ReLU', inplace=True),
    dict(cls='ConvTranspose2d', kernel_size=6, stride=3),
]


# BasicCNN https://github.com/SudeepDasari/one_shot_transformers/blob/ecd43b0c182451b67219fdbc7d6a3cd912395f17/hem/models/basic_embedding.py#L224
# aka SmallNet from https://arxiv.org/pdf/2011.05970.pdf
# which follows Fig 4 of http://www.roboticsproceedings.org/rss14/p02.pdf
# spatial softmax is applied externally
smallnet_encoder = [
    dict(cls='Conv2d', out_channels=32, kernel_size=3, stride=2, padding=1),
    dict(cls='ReLU', inplace=True),
    dict(cls='Conv2d', out_channels=32, kernel_size=3, stride=2, padding=1),
    dict(cls='ReLU', inplace=True),
    dict(cls='Conv2d', out_channels=32, kernel_size=3, stride=1, padding=1),
    dict(cls='ReLU', inplace=True),
    dict(cls='Conv2d', out_channels=32, kernel_size=3, stride=1, padding=1),
    dict(cls='ReLU', inplace=True),
]

# from https://github.com/AntixK/PyTorch-VAE
# see https://github.com/AntixK/PyTorch-VAE/blob/8700d245a9735640dda458db4cf40708caf2e77f/models/beta_vae.py#L75
antix_encoder = [
    dict(cls='Conv2d', out_channels=16, kernel_size=3, stride=2, padding=1),
    dict(cls='ReLU', inplace=True),
    dict(cls='Conv2d', out_channels=32, kernel_size=3, stride=2, padding=1),
    dict(cls='ReLU', inplace=True),
    dict(cls='Conv2d', out_channels=64, kernel_size=3, stride=2, padding=1),
    dict(cls='ReLU', inplace=True),
]
antix_decoder = [
    dict(
        cls='ConvTranspose2d',
        in_channels=64,
        out_channels=32,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
    ),
    dict(cls='ReLU', inplace=True),
    dict(
        cls='ConvTranspose2d', out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1
    ),
    dict(cls='ReLU', inplace=True),
    dict(
        cls='ConvTranspose2d', out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1
    ),
    dict(cls='ReLU', inplace=True),
    dict(cls='Conv2d', kernel_size=3, padding=1),
]


# from https://github.com/ramanans1/plan2explore/blob/3aa005a60d5b5afa74e257dfee83fb7115291394/plan2explore/networks/conv.py#L26
plan2explore_encoder = [
    dict(cls='Conv2d', out_channels=32, kernel_size=4, stride=2),
    dict(cls='ReLU', inplace=True),
    dict(cls='Conv2d', out_channels=64, kernel_size=4, stride=2),
    dict(cls='ReLU', inplace=True),
    dict(cls='Conv2d', out_channels=128, kernel_size=4, stride=2),
    dict(cls='ReLU', inplace=True),
    dict(cls='Conv2d', out_channels=256, kernel_size=4, stride=2),
    dict(cls='ReLU', inplace=True),
]
# decoder is different
plan2explore_decoder = [
    dict(cls='ConvTranspose2d', in_channels=256, out_channels=128, kernel_size=4, stride=2),
    dict(cls='ReLU', inplace=True),
    dict(cls='ConvTranspose2d', out_channels=64, kernel_size=4, stride=2),
    dict(cls='ReLU', inplace=True),
    dict(cls='ConvTranspose2d', out_channels=32, kernel_size=4, stride=2, output_padding=1),
    dict(cls='ReLU', inplace=True),
    dict(cls='ConvTranspose2d', kernel_size=4, stride=2),
]


# from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py#L51
# also used in https://github.com/facebookresearch/torchbeast/blob/3f3029cf3d6d488b8b8f952964795f451a49048f/torchbeast/monobeast.py#L545
nature_dqn_encoder = [
    dict(cls='Conv2d', out_channels=32, kernel_size=8, stride=4),
    dict(cls='ReLU', inplace=True),
    dict(cls='Conv2d', out_channels=64, kernel_size=4, stride=2),
    dict(cls='ReLU', inplace=True),
    dict(cls='Conv2d', out_channels=64, kernel_size=3, stride=1),
    dict(cls='ReLU', inplace=True),
]


# as https://github.com/facebookresearch/torchbeast/blob/3f3029cf3d6d488b8b8f952964795f451a49048f/torchbeast/polybeast_learner.py#L134
def impala_encoder(in_channels=None, **unused):
    layers = []
    for out_channels in [16, 32, 32]:
        impala_block = nn.Sequential(
            *[
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ImpalaResidual(out_channels),
                ImpalaResidual(out_channels),
            ]
        )
        layers.append(impala_block)
        in_channels = out_channels
    layers.append(nn.ReLU(inplace=True))
    return layers


class ImpalaResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # not inplace=True b/c residual
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        x = self.conv1(self.relu1(x))
        x = self.conv2(self.relu2(x))
        x += identity
        return x
