from dsb.dependencies import *

from .svea_vit.modules import SharedTransformer, HeadCNN


class ViT(nn.Module):
    def __init__(
        self,
        obs_space,
        embedding_dim=512,
        detach_embedding=False,
        #
        patch_size=8,
        depth=4,
        num_heads=8,
        mlp_ratio=1.0,
        qvk_bias=False,
        #
        # num_head_layers=0,
        # num_filters=32,
    ):
        super().__init__()
        assert len(obs_space.shape) == 3  # check if image space
        self.obs_space = obs_space
        self.embedding_dim = embedding_dim
        self.detach_embedding = detach_embedding

        self.f = SharedTransformer(
            obs_space.shape,
            patch_size,
            self.embedding_dim,
            depth,
            num_heads,
            mlp_ratio,
            qvk_bias,
        )

        # self.head = HeadCNN(shared.out_shape, num_head_layers, num_filters)

    def forward(self, x, **kwargs):
        y = self.f(x)
        return y
