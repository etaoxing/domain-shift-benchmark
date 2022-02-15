from dsb.dependencies import *

from dsb.utils import torchify, untorchify, module_repr_include, compute_output_shape
from dsb.builder import build_optim, build_network_modules

import itertools


# TODO: ortho init everything (should also include actor/critic, check again?)
class RAE(nn.Module):
    def __init__(
        self,
        obs_space,
        encoder_network_params=[],
        decoder_network_params=[],
        embedding_dim=64,  # latent dim
        encoder_optim_params=dict(cls='Adam', lr=1e-3),
        # vvv see https://github.com/denisyarats/pytorch_sac_ae/blob/74eed092e5b1a857c32aad05e2fc65f2f9add37e/train.py#L60
        decoder_optim_params=dict(cls='Adam', lr=1e-3, weight_decay=1e-7),
        optimize_interval=1,
        clip_grad_norm=None,
        detach_embedding=False,  # if True, detach so other losses will not update encoder
        #
        beta=9,
    ):
        super().__init__()
        assert len(obs_space.shape) == 3  # check if image space
        self.obs_space = obs_space
        self.input_dim = int(np.prod(obs_space.shape))
        self.embedding_dim = embedding_dim
        self.optimize_interval = optimize_interval
        self.clip_grad_norm = clip_grad_norm
        self.detach_embedding = detach_embedding

        self.beta = beta
        self.normalized_beta = self.beta * self.embedding_dim / self.input_dim
        # we use the corresponding normalized beta, ref note in neg_logprob() from beta_vae.py
        # B' = 1e-6 * 3 * 84 * 84 ~= 0.021168, since sac_ae implemenation
        # averages over the dimensions of the image
        # https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/sac_ae.py#L376
        #
        # sac_ae uses 84x84 RGB images, with latent dim 50, https://github.com/denisyarats/pytorch_sac_ae/blob/74eed092e5b1a857c32aad05e2fc65f2f9add37e/train.py#L53
        # B = B' * (3*84*84)/(50) ~= 8.962, just round to 9 as default
        # which is close to the experimental optimal found
        # on MNIST in the RAE paper?, see Appendix A https://arxiv.org/pdf/1903.12436.pdf#page=13
        # unsure though, need to double check the loss averaging and weighting
        # of https://github.com/ParthaEth/Regularized_autoencoders-RAE-/blob/3854e1ac607c150134f606806f6e5ca7a94cf1a8/models/rae/loss_functions.py#L26
        #
        # decoder_latent_beta=1e-6, # from https://github.com/denisyarats/pytorch_sac_ae/blob/74eed092e5b1a857c32aad05e2fc65f2f9add37e/train.py
        # also see section B.5 of https://arxiv.org/pdf/1910.01741.pdf#page=13
        # decoder_latent_lambda is actually beta from equation 11 in
        # the original RAE paper, https://arxiv.org/pdf/1903.12436.pdf#page=4,
        # lambda is the weight decay

        in_channels = self.obs_space.shape[0]

        self.encoder = build_network_modules(encoder_network_params, in_channels=in_channels)
        self.encoder = nn.Sequential(*self.encoder)

        self.conv_output_shape, n_flatten = compute_output_shape(
            self.obs_space.sample(), self.encoder
        )

        self.fc_encoder = nn.Sequential(
            *[
                nn.Linear(n_flatten, self.embedding_dim),
                # vvv as https://github.com/denisyarats/pytorch_sac_ae/blob/74eed092e5b1a857c32aad05e2fc65f2f9add37e/encoder.py#L32
                nn.LayerNorm(self.embedding_dim),
                nn.Tanh(),
            ]
        )

        self.fc_decoder = nn.Linear(self.embedding_dim, n_flatten)

        self.decoder = build_network_modules(decoder_network_params, out_channels=in_channels)
        self.decoder = nn.Sequential(*self.decoder)

        _encoder_params = itertools.chain(self.encoder.parameters(), self.fc_encoder.parameters())
        self.encoder_optimizer = build_optim(encoder_optim_params, params=_encoder_params)

        _decoder_params = itertools.chain(self.fc_decoder.parameters(), self.decoder.parameters())
        self.decoder_optimizer = build_optim(decoder_optim_params, params=_decoder_params)

    def forward(self, x, with_conv_output=False, detach_embedding=None):
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

    def reconstruct(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def encode(self, x):
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        z = self.fc_encoder(h)
        return z

    def decode(self, z):
        z = self.fc_decoder(z)
        z = z.view(-1, *self.conv_output_shape)
        recon = self.decoder(z)
        return recon

    def neg_logprob(self, recon, x):
        B = x.size(0)
        return F.mse_loss(recon, x, reduction='sum') / B

    def optimize(self, x, embedding_target=None):
        assert embedding_target is not None  # use GlowPixelNoiseNormalizer
        if embedding_target is None:
            embedding_target = x

        opt_info = {}

        z = self.encode(x)
        recon = self.decode(z)

        recon_loss = self.neg_logprob(recon, embedding_target.detach())
        # from https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/sac_ae.py#L378
        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * z.pow(2).sum(1)).mean()

        loss = recon_loss + self.normalized_beta * latent_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        opt_info['embedding_head_loss'] = loss.item()
        opt_info['recon_loss'] = recon_loss.item()
        opt_info['latent_loss'] = latent_loss.item()
        if self.clip_grad_norm is not None:
            opt_info['embedding_grad_norm'] = grad_norm.item()

        # TODO: compute SSIM b/w x and recon?
        # https://github.com/VainF/pytorch-msssim

        return opt_info

    @property
    def opt_info_keys(self):
        k = ['embedding_head_loss', 'recon_loss', 'latent_loss']
        if self.clip_grad_norm is not None:
            k.append('embedding_grad_norm')
        return k

    def state_dict(self, *args, **kwargs):
        state_dict = dict(
            model=super().state_dict(*args, **kwargs),
            encoder_optimizer=self.encoder_optimizer.state_dict(),
            decoder_optimizer=self.decoder_optimizer.state_dict(),
        )
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict['model'])
        self.encoder_optimizer.load_state_dict(state_dict['encoder_optimizer'])
        self.decoder_optimizer.load_state_dict(state_dict['decoder_optimizer'])

    def __repr__(self):
        s = module_repr_include(
            super().__repr__(),
            dict(
                beta=self.beta,
                normalized_beta=self.normalized_beta,
            ),
        )
        return s
