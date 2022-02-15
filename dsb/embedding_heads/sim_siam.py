from dsb.dependencies import *

from dsb.utils import torchify, untorchify, compute_output_shape
from dsb.agents.utils import update_target_network
import dsb.builder as builder
from dsb.builder import module_reset_parameters


# original ref: https://github.com/facebookresearch/simsiam/blob/a7bc1772896d0dad0806c51f0bb6f3b16d290468/simsiam/builder.py
# also https://github.com/PatrickHua/SimSiam
class SimSiam(nn.Module):
    # SimSiam is just BYOL w/o momentum encoder
    #
    # Training tip: Should see embedding_loss decrease quickly to -0.7 to -0.9 range.
    # If it reaches -1 immediately, then model has probably diverged.
    # If the loss is oscillating around the -0.4 to -0.6 range then try increasing
    # embedding_dim.
    # See Figure 2 of SimSiam.

    def __init__(
        self,
        obs_space,
        encoder_network_params=[],
        embedding_dim=2048,  # latent dim
        # optim_params=dict(cls='Adam', lr=3e-4, weight_decay=1e-4), # NOTE: SimSiam uses SGD w/ momentum, weight decay, & cosine decay schedule
        optim_params=dict(cls='SGD', lr=0.05, weight_decay=1e-4, momentum=0.9),
        optimize_interval=1,
        detach_embedding=False,  # if True, detach so other losses will not update encoder
        #
        aug_params=None,
        detach_augmented=True,
        extra_aug_params=None,
        forward_aug_params='same',
        prediction_head_bottleneck_dim=512,  # see appendix B of https://arxiv.org/pdf/2011.10566.pdf
        symmetric=True,
        use_target_for_pair=False,  # if True, then use momentum encoder, so becomes BYOL
        tau=0.005,
    ):
        super().__init__()
        assert len(obs_space.shape) == 3  # check if image space
        self.obs_space = obs_space
        self.embedding_dim = embedding_dim
        self.detach_embedding = detach_embedding
        self.detach_augmented = detach_augmented

        self.symmetric = symmetric
        self.use_target_for_pair = use_target_for_pair
        self.tau = tau

        in_channels = self.obs_space.shape[0]
        img_size = (self.obs_space.shape[1], self.obs_space.shape[2])

        self.aug = builder.build_aug(aug_params, img_size=img_size)
        # TODO: extra_aug applied on top of aug, so change img_size
        self.extra_aug = builder.build_aug(extra_aug_params, img_size=img_size)

        # TODO: add flag to change forward_aug in train vs. eval
        if forward_aug_params == 'same':
            self.forward_aug = self.aug
        else:
            self.forward_aug = builder.build_aug(forward_aug_params, img_size=img_size)

        self.encoder = builder.build_network_modules(
            encoder_network_params, in_channels=in_channels
        )
        self.encoder = nn.Sequential(*self.encoder)

        self.conv_output_shape, n_flatten = compute_output_shape(
            self.obs_space.sample(), self.encoder, aug=self.aug
        )
        # TODO: check that extra_aug doesn't change shape?

        # NOTE:
        # SimSiam follows SimCLR and discards f for downstream tasks?
        # for now we just give the projection_head as output.
        # also check out https://arxiv.org/pdf/2010.10241.pdf
        # and https://untitled-ai.github.io/appendix-for-understanding-self-supervised-contrastive-learning.html
        # for replacing batchnorm ideas

        # https://github.com/facebookresearch/simsiam/blob/a7bc1772896d0dad0806c51f0bb6f3b16d290468/simsiam/builder.py#L26
        self.projection_head = nn.Sequential(
            *[  # f
                nn.Linear(n_flatten, embedding_dim, bias=False),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, embedding_dim, bias=False),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, embedding_dim, bias=False),
                nn.BatchNorm1d(embedding_dim, affine=False),  # see section 4.4 of SimSiam
            ]
        )

        self.prediction_head = nn.Sequential(
            *[  # h
                nn.Linear(embedding_dim, prediction_head_bottleneck_dim, bias=False),
                nn.BatchNorm1d(prediction_head_bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Linear(prediction_head_bottleneck_dim, embedding_dim),
            ]
        )

        self.optimizer = builder.build_optim(optim_params, params=self.parameters())
        self.optimize_interval = optimize_interval

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.apply(module_reset_parameters)
        self.projection_head.apply(module_reset_parameters)
        self.prediction_head.apply(module_reset_parameters)

        self._create_target_networks()

    def _create_target_networks(self):
        if self.use_target_for_pair:
            self.encoder_target = copy.deepcopy(self.encoder)
            self.encoder_target.load_state_dict(self.encoder.state_dict())

            self.projection_head_target = copy.deepcopy(self.projection_head)
            self.projection_head_target.load_state_dict(self.projection_head.state_dict())

    def forward(self, x, with_conv_output=False, detach_embedding=None):
        if self.forward_aug:
            if self.detach_augmented:
                with torch.no_grad():
                    x = self.forward_aug(x)
            else:
                x = self.forward_aug(x)

        z = self.encode(x)

        detach_embedding = (
            detach_embedding if detach_embedding is not None else self.detach_embedding
        )
        if detach_embedding:
            z = z.detach()

        if with_conv_output:
            raise NotImplementedError
        else:
            return z

    def encode(self, x):
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def _encode_target(self, x):
        h = self.encoder_target(x)
        h = h.flatten(start_dim=1)
        z = self.projection_head_target(h)
        return z

    def D(self, p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()  # note the stop_gradient

        # BYOL also does, see https://github.com/lucidrains/byol-pytorch/blob/8efcc905d565b6ca33a9c7d814cb0687bc06a282/byol_pytorch/byol_pytorch.py#L40
        # and https://github.com/astooke/rlpyt/blob/b05f954e88fc774d61c6504ebe62ff71a181ad7a/rlpyt/ul/algos/ul_for_rl/augmented_temporal_similarity.py#L142

    def optimize(self, x, embedding_target=None):
        assert embedding_target is None
        opt_info = {}

        # augmentations should be different b/w views and elements in batch
        if self.detach_augmented:
            with torch.no_grad():
                x1, x2 = self.aug(x), self.aug(x)
        else:
            x1, x2 = self.aug(x), self.aug(x)

        if self.symmetric:
            z1, z2 = self.encode(x1), self.encode(x2)
            p1, p2 = self.prediction_head(z1), self.prediction_head(z2)

            if self.use_target_for_pair:  # BYOL
                with torch.no_grad():
                    z1_target = self._encode_target(x1)
                    z2_target = self._encode_target(x2)

                loss = 0.5 * (self.D(p1, z2_target), self.D(p2, z1_target))
            else:  # SimSiam
                loss = 0.5 * (self.D(p1, z2) + self.D(p2, z1))
        else:
            # SODA: https://github.com/nicklashansen/dmcontrol-generalization-benchmark/blob/ee658ceb449b884812149b922035197be8e28c87/src/algorithms/soda.py#L40
            # also see https://arxiv.org/pdf/2007.05929.pdf and https://arxiv.org/pdf/2007.04309.pdf

            if self.extra_aug:
                if self.detach_augmented:
                    with torch.no_grad():
                        x1 = self.extra_aug(x1)
                else:
                    x1 = self.extra_aug(x1)

            z1 = self.encode(x1)  # x1 is aug_x

            if self.use_target_for_pair:
                with torch.no_grad():
                    z2 = self._encode_target(x2)
            else:
                z2 = self.encode(x2)

            p1 = self.prediction_head(z1)
            # h1 = F.normalize(p1, p=2, dim=1)
            # h2 = F.normalize(z2, p=2, dim=1)
            # loss = F.mse_loss(h1, h2.detach())
            loss = self.D(p1, z2.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_target_for_pair:
            update_target_network(self.encoder, self.encoder_target, tau=self.tau)
            update_target_network(self.projection_head, self.projection_head_target, tau=self.tau)

        opt_info['embedding_head_loss'] = loss.item()

        # see section 4.1, https://arxiv.org/pdf/2011.10566.pdf#page=3
        output_std = torch.std(F.normalize(z1.detach(), dim=1), dim=1, unbiased=True)
        opt_info['output_std'] = untorchify(output_std.mean(dim=0))
        return opt_info

    @property
    def opt_info_keys(self):
        return ['embedding_head_loss']

    def state_dict(self, *args, **kwargs):
        state_dict = dict(
            model=super().state_dict(*args, **kwargs),
            optimizer=self.optimizer.state_dict(),
        )
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict.pop('optimizer'))
        super().load_state_dict(state_dict['model'])
