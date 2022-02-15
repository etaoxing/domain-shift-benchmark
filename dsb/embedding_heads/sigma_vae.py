from dsb.dependencies import *

from dsb.utils import torchify, untorchify, compute_output_shape
from dsb.builder import build_optim, build_network_modules


def softclip(tensor, min):
    """Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials"""
    result_tensor = min + F.softplus(tensor - min)
    return result_tensor


def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)


# see https://github.com/orybkin/sigma-vae-pytorch/blob/b561fc92355e81512a81104cb1e21b06cc9459e2/model.py#L30
class SigmaVAE(nn.Module):
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
        optimal_sigma=False,
        decoder_activation_params=dict(cls='Tanh'),
    ):
        super().__init__()
        assert len(obs_space.shape) == 3  # check if image space
        self.obs_space = obs_space
        self.input_dim = int(np.prod(obs_space.shape))  # this includes channel
        self.embedding_dim = embedding_dim
        self.optimize_interval = optimize_interval
        self.clip_grad_norm = clip_grad_norm
        self.detach_embedding = detach_embedding

        self.optimal_sigma = optimal_sigma
        self.log_sigma = 0
        if not self.optimal_sigma:
            # self.log_sigma = torch.nn.Parameter(torch.full((1,), 0)[0], requires_grad=True)
            # vvv should be equivalent to line above
            self.log_sigma = torch.nn.Parameter(torch.tensor(0.0))

        in_channels = self.obs_space.shape[0]

        self.encoder = build_network_modules(encoder_network_params, in_channels=in_channels)
        self.encoder = nn.Sequential(*self.encoder)

        self.conv_output_shape, n_flatten = compute_output_shape(
            self.obs_space.sample(), self.encoder
        )

        # NOTE: could combine these layers and split output
        self.fc_mu = nn.Linear(n_flatten, self.embedding_dim)
        self.fc_var = nn.Linear(n_flatten, self.embedding_dim)

        self.fc_decoder = nn.Linear(self.embedding_dim, n_flatten)

        self.decoder = build_network_modules(decoder_network_params, out_channels=in_channels)
        self.decoder += build_network_modules([decoder_activation_params])
        self.decoder = nn.Sequential(*self.decoder)

        self.optimizer = build_optim(optim_params, params=self.parameters())

    def forward(self, x, with_conv_output=False, detach_embedding=None):
        conv_output, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        detach_embedding = (
            detach_embedding if detach_embedding is not None else self.detach_embedding
        )
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
        logvar = self.fc_var(h)
        return conv_output, mu, logvar

    def decode(self, z):
        z = self.fc_decoder(z)
        z = z.view(-1, *self.conv_output_shape)
        recon = self.decoder(z)
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
        # https://github.com/orybkin/sigma-vae-pytorch/blob/b561fc92355e81512a81104cb1e21b06cc9459e2/model.py#L143
        # doesn't average by batch dim
        # return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # average by batch dim
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def neg_logprob(self, recon, x):
        if self.optimal_sigma:
            log_sigma = ((x - recon) ** 2).mean([0, 1, 2, 3], keepdim=True).sqrt().log()
            self.log_sigma = log_sigma.item()
        else:
            # Sigma VAE learns the variance of the decoder as another parameter
            log_sigma = self.log_sigma

        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        log_sigma = softclip(log_sigma, -6)
        # probably don't need to softclip if not self.optimal_sigma?
        # "We also observe that this clipping is unnecessary when learning a shared σ value.""

        rec = gaussian_nll(recon, log_sigma, x).sum()
        return rec / x.size(0)

    def optimize(self, x, embedding_target=None):
        if embedding_target is None:
            embedding_target = x

        opt_info = {}

        _, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        recon_loss = self.neg_logprob(recon, embedding_target.detach())
        kld_loss = self.kl_divergence(mu, logvar)
        loss = recon_loss + kld_loss

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
        super().load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
