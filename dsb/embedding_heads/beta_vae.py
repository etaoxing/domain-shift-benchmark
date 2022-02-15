from re import I
from dsb.dependencies import *

from dsb.utils import torchify, untorchify, module_repr_include, compute_output_shape
import dsb.builder as builder
from dsb.builder import module_reset_parameters


class BetaVAE(nn.Module):
    def __init__(
        self,
        obs_space,
        encoder_network_params=[],
        decoder_network_params=[],
        embedding_dim=64,  # latent dim
        optim_params=dict(cls='Adam', lr=1e-3),
        optimize_interval=1,
        clip_grad_norm=None,
        detach_embedding=False,  # if True, detach so other losses will not update encoder
        #
        beta=1,  # NOTE: this is the unnormalized beta, we compute normalized_beta below. standard VAE has normalized_beta=1
        min_var=None,
        decoder_distribution='bernoulli',
    ):
        super().__init__()
        assert len(obs_space.shape) == 3  # check if image space
        self.obs_space = obs_space
        self.input_dim = int(np.prod(obs_space.shape))  # this includes channel
        self.embedding_dim = embedding_dim
        self.optimize_interval = optimize_interval
        self.clip_grad_norm = clip_grad_norm
        self.detach_embedding = detach_embedding

        self.beta = beta
        # see figure 6 and the 'Understanding the effects of \beta' section
        # from https://openreview.net/pdf?id=Sy2fzU9gl#page=9
        self.normalized_beta = self.beta * self.embedding_dim / self.input_dim

        self.min_logvar = np.log(min_var) if min_var is not None else None
        self.decoder_distribution = decoder_distribution
        # NOTE: use sigmoid for bernoulli, tanh for gaussian.
        # input should also be rescaled accordingly based on normalizer
        # so [0, 1] for bernoulli and [-1, 1] for gaussian
        if self.decoder_distribution == 'bernoulli':
            decoder_activation_params = dict(cls='Sigmoid')
        elif self.decoder_distribution == 'gaussian':
            decoder_activation_params = dict(cls='Tanh')
        else:
            raise ValueError

        in_channels = self.obs_space.shape[0]

        self.encoder = builder.build_network_modules(
            encoder_network_params, in_channels=in_channels
        )
        self.encoder = nn.Sequential(*self.encoder)

        self.conv_output_shape, n_flatten = compute_output_shape(
            self.obs_space.sample(), self.encoder
        )

        # NOTE: could combine these layers and split output
        self.fc_mu = nn.Linear(n_flatten, self.embedding_dim)
        self.fc_var = nn.Linear(n_flatten, self.embedding_dim)

        self.fc_decoder = nn.Linear(self.embedding_dim, n_flatten)

        self.decoder = builder.build_network_modules(
            decoder_network_params, out_channels=in_channels
        )
        self.decoder += builder.build_network_modules([decoder_activation_params])
        self.decoder = nn.Sequential(*self.decoder)

        self.optimizer = builder.build_optim(optim_params, params=self.parameters())

    def reset_parameters(self):
        self.encoder.apply(module_reset_parameters)
        self.fc_mu.reset_parameters()
        self.fc_var.reset_parameters()
        self.fc_decoder.reset_parameters()
        self.decoder.apply(module_reset_parameters)

    def forward(self, x, with_conv_output=False, detach_embedding=None):
        conv_output, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        detach_embedding = self.detach_embedding if detach_embedding is None else detach_embedding
        if detach_embedding:
            z = z.detach()

        if with_conv_output:
            return z, conv_output
        else:
            return z

    def reconstruct(self, x):
        _, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon

    def encode(self, x):
        conv_output = self.encoder(x)
        h = conv_output.flatten(start_dim=1)
        mu = self.fc_mu(h)

        if self.min_logvar is None:
            logvar = self.fc_var(h)
        else:
            logvar = self.min_logvar + torch.abs(self.fc_var(h))
        return conv_output, mu, logvar

    def decode(self, z):
        z = self.fc_decoder(z)
        z = z.view(-1, *self.conv_output_shape)
        recon = self.decoder(z)
        if self.decoder_distribution == 'bernoulli':
            pass
        elif self.decoder_distribution == 'gaussian':
            pass
        # elif self.decoder_distribution == 'gaussian_identity_variance':
        #     recon = torch.clamp(recon, 0, 1)
        else:
            raise ValueError
        return recon

    def reparameterize(self, mu, logvar):
        if self.training:  # TODO: check if network is frozen?
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # don't add noise for eval
            # see https://github.com/vitchyr/rlkit/blob/86db9c2bab51352b95f3f1acba19d6173f5feadd/rlkit/torch/vae/vae_base.py#L107
            # and https://github.com/thanard/hallucinative-topological-memory/blob/19c6bab4f09e402753b3a5c9015b6f928cb7f64b/eval.py#L102
            z = mu
        return z

    def kl_divergence(self, mu, logvar):
        # doesn't average by batch dim
        # return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # average by batch dim
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def neg_logprob(self, recon, x):
        B = x.size(0)
        if self.decoder_distribution == 'bernoulli':
            # https://github.com/thanard/hallucinative-topological-memory/blob/82f63f01e7b6b552d515275249d5a11a5be6fe0a/models.py#L320
            # https://github.com/vitchyr/rlkit/blob/v0.1.2/rlkit/torch/vae/conv_vae.py#L231
            recon = recon.view(-1, self.input_dim)
            x = x.view(-1, self.input_dim)

            # equivalent to l = F.binary_cross_entropy(recon, x) * self.input_dim
            l = F.binary_cross_entropy(recon, x, reduction='sum')
        elif self.decoder_distribution == 'gaussian':
            # elif self.decoder_distribution == 'gaussian' or self.decoder_distribution == 'gaussian_identity_variance':
            l = F.mse_loss(recon, x, reduction='sum')
            # as https://github.com/AntixK/PyTorch-VAE/blob/master/models/betatc_vae.py#L163

            # NOTE: many implementations also average across (channel, height, width)
            # this affects the reported beta scaling with the KL term.
            # let b be the reported constant, B = unnormalized beta, B' = normalized beta, m = latent dim, n = input dim.
            # we use B' to weight our KL term, since our reconstruction sums over the image dims and only averages over batch dim
            #
            # for instance with SkewFit, see 'Visual Pickup' of Table 3 in https://arxiv.org/pdf/1903.03698.pdf#page=19,
            # they report using b=30 and m=16, this correponds to B=30/16,
            # since they use F.mse_loss(...), which averages across (channel, height, width),
            # https://github.com/vitchyr/rlkit/blob/86db9c2bab51352b95f3f1acba19d6173f5feadd/rlkit/torch/vae/conv_vae.py#L242
            # and with n=(3, 48, 48), they have B'=((30/16)*16)/(3*48*48) ~= 0.00434
            #
            # similarly:
            # https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/sac_ae.py#L376
            # https://github.com/AntixK/PyTorch-VAE/blob/8700d245a9735640dda458db4cf40708caf2e77f/models/beta_vae.py#L139
            #
            # as another example, consider the VAE baseline in ATC
            # from https://github.com/astooke/rlpyt/blob/b05f954e88fc774d61c6504ebe62ff71a181ad7a/rlpyt/ul/experiments/ul_for_rl/configs/atari/atari_vae.py
            # with b=1, m=256, n=(3, 84, 84).
            # this correponds to B'=1/3=0.333 since their recon loss
            # sums over (height, width) and averages over (channel)
            # https://github.com/astooke/rlpyt/blob/b05f954e88fc774d61c6504ebe62ff71a181ad7a/rlpyt/ul/algos/ul_for_rl/vae.py#L164
            # which is equivalent to B=(1/3)*(3*84*84)/(256)=27.5625
            # if we take b=0.1 as per section A.4 of https://arxiv.org/pdf/2009.08319.pdf#page=18
            # then we get B'=0.0333 and B=2.75625
            #
            # others which don't average over all dims:
            # https://github.com/YannDubs/disentangling-vae/blob/474fc4672c8f54f9403dd0e615ec64485dada778/disvae/models/losses.py#L392
            #
            # https://github.com/wilson1yan/contrastive-forward-model/blob/25cc75fdf6e81dc6a925024005f784e332eb8714/cfm/baselines/train_planet.py#L40
            #
            # https://github.com/orybkin/sigma-vae-pytorch/blob/b561fc92355e81512a81104cb1e21b06cc9459e2/model.py#L135
            # (though as per their note, gaussian_nll() sums), also reference 'Loss implementation details' section
            # from their paper https://arxiv.org/pdf/2006.13202.pdf#page=4
        else:
            raise ValueError
        return l / B

    def optimize(self, x, embedding_target=None):
        if embedding_target is None:
            embedding_target = x

        opt_info = {}

        _, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        recon_loss = self.neg_logprob(recon, embedding_target.detach())
        kld_loss = self.kl_divergence(mu, logvar)
        loss = recon_loss + self.normalized_beta * kld_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        opt_info['embedding_head_loss'] = loss.item()
        opt_info['recon_loss'] = recon_loss.item()
        opt_info['kld_loss'] = kld_loss.item()
        if self.clip_grad_norm is not None:
            opt_info['embedding_grad_norm'] = grad_norm.item()

        return opt_info

    @property
    def opt_info_keys(self):
        k = ['embedding_head_loss', 'recon_loss', 'kld_loss']
        if self.clip_grad_norm is not None:
            k.append('embedding_grad_norm')
        return k

    def state_dict(self, *args, **kwargs):
        state_dict = dict(
            model=super().state_dict(*args, **kwargs),
            optimizer=self.optimizer.state_dict(),
        )
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict.pop('optimizer'))
        super().load_state_dict(state_dict['model'])

    def __repr__(self):
        s = module_repr_include(
            super().__repr__(),
            dict(
                beta=self.beta,
                normalized_beta=self.normalized_beta,
            ),
        )
        return s
