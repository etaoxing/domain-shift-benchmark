from dsb.dependencies import *

# Input x to normalizers should be torch.FloatTensor
# Images are given in range [0,255]


class Normalizer(nn.Module):
    @property
    def requires_update(self):  # change to True if need to call update_stats
        return False

    def update_stats(self, x):
        raise NotImplementedError

    # these should work with any num of batch dimensions, like
    # (B, ...), (B, T, ...), (B, T, N, ...) and so on
    # B = batch, T = time / trajectory, N = framestack
    def forward(self, x, inv_norm=False):
        if inv_norm:
            return self.inv_normalize(x)
        else:
            return self.normalize(x)

    def normalize(self, x):
        raise NotImplementedError

    def inv_normalize(self, x):
        raise NotImplementedError


class IdentityNormalizer(Normalizer):
    def __init__(self, obs_space):
        super().__init__()

    @property
    def requires_update(self):
        return False

    def update_stats(self, x):
        pass

    def forward(self, x, **kwargs):
        return x


class PixelRescaleNormalizer(Normalizer):
    """
    if zero_one_range==True, rescales images in range [0,255] to [0,1]
    otherwise rescales images to [-1,1]
    """

    def __init__(self, obs_space, zero_one_range=False):
        super().__init__()
        self.zero_one_range = zero_one_range

    @property
    def requires_update(self):
        return False

    def update_stats(self, x):
        pass

    def normalize(self, x):
        x = x / 255.0
        if not self.zero_one_range:
            x = 2.0 * x - 1.0
        return x

    def inv_normalize(self, x):
        if not self.zero_one_range:
            x = (x + 1.0) / 2.0
        x = x * 255.0
        return x


# from https://github.com/ramanans1/plan2explore/blob/887a3ac70d1ff08ab292911a74064ca1ac5512e6/plan2explore/tools/preprocess.py#L22
# also https://github.com/denisyarats/pytorch_sac_ae/blob/74eed092e5b1a857c32aad05e2fc65f2f9add37e/utils.py#L56
class GlowPixelNoiseNormalizer(Normalizer):
    """
    if half_zero_one_range==True, rescales images in range[-0.5, 0.5]
    otherwise rescales images to [-1,1]
    """

    def __init__(self, obs_space, bits=5, half_zero_one_range=True):
        super().__init__()
        self.bits = bits
        self.bins = 2 ** bits
        self.half_zero_one_range = half_zero_one_range

    @property
    def requires_update(self):
        return False

    def update_stats(self, x):
        pass

    def normalize(self, x):
        if self.bits < 8:
            x = torch.floor(x / 2 ** (8 - self.bits))
        x = x / self.bins
        x = x + torch.rand_like(x) / self.bins  # uniform noise
        x = x - 0.5
        if not self.half_zero_one_range:
            x = x * 2.0
        return x

    def inv_normalize(self, x):
        if not self.half_zero_one_range:
            x = x / 2.0
        x = x + 0.5
        x = torch.floor(self.bins * x)
        x = x * (256.0 / self.bins)  # 2**8 == 256
        x = torch.clamp(x, 0, 255)
        return x


# from https://github.com/astooke/rlpyt/blob/master/rlpyt/models/running_mean_std.py
# np version https://github.com/spitis/mrl/blob/e40f8ae5b453c38a1d968d2a4da9d26f78e22b4e/mrl/modules/normalizer.py
# refs:
# https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
# https://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation/103025#103025
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
class RunningMeanStdNormalizer(Normalizer):
    def __init__(self, obs_space, rescale_pixels=False, norm_clip=5, min_var=1e-6):
        super().__init__()
        self.obs_space = obs_space
        self.shape = self.obs_space.shape
        self.rescale_pixels = rescale_pixels
        self.norm_clip = norm_clip
        self.min_var = min_var

        self.register_buffer("mean", torch.zeros(self.shape))
        self.register_buffer("var", torch.ones(self.shape))
        self.register_buffer("count", torch.zeros(()))

        self._requires_update = True

    @property
    def requires_update(self):
        return self._requires_update

    @requires_update.setter
    def requires_update(self, val):
        self._requires_update = val

    def update_stats(self, x):
        if self.rescale_pixels:
            x = x / 255.0

        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = len(x)

        if self.count == 0:
            self.mean[:] = batch_mean
            self.var[:] = batch_var
        else:
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean[:] = self.mean + delta * batch_count / total
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
            self.var[:] = M2 / total
        self.count += batch_count

    def normalize(self, x):
        if self.rescale_pixels:
            x = x / 255.0

        std = self.std
        x = (x - self.mean) / std
        if self.norm_clip:
            x = torch.clamp(x, -self.norm_clip, self.norm_clip)
        return x

    def inv_normalize(self, x):  # https://github.com/pytorch/vision/issues/848
        std = self.std
        x = x * std + self.mean
        # x = (x + (self.mean / std)) / (1 / std) # div by zero should be covered by min_var

        x = torch.clamp(x, 0.0, 1.0)

        if self.rescale_pixels:
            x = x * 255.0
        return x

    @property
    def std(self):
        var = self.var.clone().detach()
        if self.min_var:
            var = torch.clamp(var, min=self.min_var)
        return var.sqrt()

    def __repr__(self):
        s = f'RunningMeanStdNormalizer(in_features={self.shape})'
        return s


class DictNormalizer(nn.Module):
    def __init__(self, normalizers, normalizers_params={}):
        super().__init__()
        self.normalizers = nn.ModuleDict(normalizers)
        self.normalizers_params = normalizers_params

        self._requires_update = {
            k: v.requires_update and normalizers_params[k].get('update_with', True)
            for k, v in self.normalizers.items()
        }

    @property
    def requires_update(self):
        return self._requires_update

    def update_stats(self, x):
        for k in x.keys():
            if k in self.normalizers.keys():
                if self._requires_update[k]:
                    self.normalizers[k].update_stats(x[k])

    def forward(self, x, which=None, **kwargs):
        if which is not None:
            return self.normalizers[which](x, **kwargs)
        else:
            y = dict()
            for k, x_ in x.items():
                if k.startswith('c_'):
                    k_ = k[2:]
                    y[k] = self.normalizers[k_](x_, **kwargs)
                elif k.startswith('fs_'):
                    # TODO: these normalizers should all work with the extra dim with without
                    # reshaping, so remove?
                    k_ = k[3:]
                    x_s = tuple(x_.shape)
                    B, T = x_s[0], x_s[1]
                    x__ = x_.view((B * T,) + x_s[2:])
                    o = self.normalizers[k_](x__, **kwargs)
                    o = o.reshape((B, T) + o.shape[1:])
                    y[k] = o
                else:
                    if k in self.normalizers.keys():
                        y[k] = self.normalizers[k](x_, **kwargs)
                    else:
                        y[k] = x_
            return y

    def __repr__(self):
        s = 'DictNormalizer(\n'
        for k, v in self.normalizers.items():
            np = self.normalizers_params[k] if k in self.normalizers_params.keys() else {}
            s += f'  ({k}): {v}, {hex(id(v))}, {np}\n'
        s += ')'
        return s
