import torch
import torch.nn as nn

try:
    import kornia.augmentation as aug

    # generally, kornia.augmentation expects images in range [0, 255],
    # but it doesn't matter for RandomShift and Intensity.
    # taken from https://arxiv.org/pdf/2004.13649.pdf#page=22

    # basically https://github.com/astooke/rlpyt/blob/b05f954e88fc774d61c6504ebe62ff71a181ad7a/rlpyt/ul/algos/ul_for_rl/augmented_temporal_contrast.py#L127
    class RandomShift(nn.Module):
        def __init__(self, img_size=None, pad=4, **unused):
            super().__init__()
            assert len(img_size) == 2
            self.pad = nn.ReplicationPad2d(pad)
            self.crop = aug.RandomCrop(img_size, align_corners=True)
            # align_corners discussion
            # https://github.com/kornia/kornia/pull/546
            # https://github.com/kornia/kornia/issues/715

        def forward(self, x):
            return self.crop(self.pad(x))

    # temp wrapper for now
    # track https://github.com/kornia/kornia/pull/987
    class Normalize(aug.Normalize):
        """
        verify that matches torchvision

        >>> import torch
        >>> x = torch.randn(1, 3, 224, 224)
        >>> mean = [0.485, 0.456, 0.406]
        >>> std = [0.229, 0.224, 0.225]
        >>> import kornia.augmentation as aug
        >>> a = aug.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        >>> from torchvision import transforms as T
        >>> t = T.Normalize(mean=mean, std=std)
        >>> y1 = a(x)
        >>> y2 = t(x)
        >>> assert torch.isclose(y1, y2).all()
        """

        def __init__(self, *args, img_size=None, mean=None, std=None, **kwargs):
            mean = torch.tensor(mean)
            std = torch.tensor(std)
            super().__init__(*args, mean=mean, std=std, **kwargs)


except ImportError as e:
    print(e)


class Intensity(nn.Module):
    def __init__(self, scale=0.1, img_size=None):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


class RepeatGrayscale(nn.Module):
    def __init__(self, img_size=None):
        super().__init__()

    def forward(self, x):
        assert x.shape[1] == 1
        return x.repeat((1, 3, 1, 1))


class RandomConv(nn.Module):
    def __init__(self, img_size=None):
        super().__init__()

    # from https://github.com/nicklashansen/dmcontrol-generalization-benchmark/blob/ee658ceb449b884812149b922035197be8e28c87/src/augmentations.py#L68
    def forward(self, x):
        """Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
        if len(x.shape) == 3:
            x = x.view(1, -1)
            batched = False
        else:
            batched = True

        n, c, h, w = x.shape
        total_out = []
        for i in range(n):
            weights = torch.randn(3, 3, 3, 3).to(x.device)
            temp_x = x[i : i + 1].reshape(-1, 3, h, w) / 255.0
            temp_x = nn.functional.pad(temp_x, pad=[1] * 4, mode='replicate')
            out = torch.sigmoid(nn.functional.conv2d(temp_x, weights)) * 255.0
            total_out.append(out)
        total_out = torch.cat(total_out, axis=0)

        if batched:
            total_out = total_out.reshape(n, c, h, w)
        else:
            total_out = total_out.reshape(c, h, w)
        return total_out
