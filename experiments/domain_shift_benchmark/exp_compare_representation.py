from dsb.utils import update_cfg

from expshared import (
    _EMBEDDING_DIM,
    _POLICY_OPTIM,
    _BC,
    _EMBEDDING_HEAD,
)

_EXP_DIR = './workdir_experiments/dsb/exp_compare_representation/'


_impala_encoder = dict(
    ckpt_dir=_EXP_DIR + 'impala_encoder/',
    state_normalizer=dict(
        cls='DictNormalizer',
        observation=dict(cls='RunningMeanStdNormalizer'),
        achieved_goal=dict(cls='PixelRescaleNormalizer', zero_one_range=True),
        desired_goal=dict(which='achieved_goal', update_with=False),  # use same normalizer
    ),
    embedding_head_wrappers=[dict(cls='SiameseGCEmbeddingHeadWrapper')],
    embedding_head=dict(
        cls='BasicCNN',
        encoder_network_params='impala_encoder',
        embedding_dim=_EMBEDDING_DIM,
        output_activation_params=None,
    ),
    **_BC,
)


_beta_vae = dict(
    ckpt_dir=_EXP_DIR + 'beta_vae/',
    state_normalizer=dict(
        cls='DictNormalizer',
        observation=dict(cls='RunningMeanStdNormalizer'),
        achieved_goal=dict(cls='PixelRescaleNormalizer', zero_one_range=False),  # [-1, 1]
        desired_goal=dict(which='achieved_goal', update_with=False),  # use same normalizer
    ),
    embedding_head_wrappers=[dict(cls='SiameseGCEmbeddingHeadWrapper')],
    embedding_head=dict(
        cls='BetaVAE',
        detach_embedding=True,  # do not have agent loss update encoder
        encoder_network_params='plan2explore_encoder',
        decoder_network_params='plan2explore_decoder',
        embedding_dim=_EMBEDDING_DIM,
        beta=10,
        decoder_distribution='gaussian',
    ),
    **_BC,
)


_beta_vae_nodetach = update_cfg(
    _beta_vae,
    dict(
        ckpt_dir=_EXP_DIR + 'beta_vae_nodetach/',
        embedding_head=dict(detach_embedding=False),
    ),
)


_rae = dict(
    ckpt_dir=_EXP_DIR + 'rae/',
    state_normalizer=dict(
        cls='DictNormalizer',
        observation=dict(cls='RunningMeanStdNormalizer'),
        achieved_goal=dict(cls='PixelRescaleNormalizer', zero_one_range=False),  # [-1, 1]
        embedding_target=dict(cls='GlowPixelNoiseNormalizer', half_zero_one_range=True),
        desired_goal=dict(which='achieved_goal', update_with=False),  # use same normalizer
    ),
    embedding_head_wrappers=[dict(cls='SiameseGCEmbeddingHeadWrapper')],
    embedding_head=dict(
        cls='RAE',
        detach_embedding=True,  # do not have agent loss update encoder
        encoder_network_params='plan2explore_encoder',
        decoder_network_params='plan2explore_decoder',
        embedding_dim=_EMBEDDING_DIM,
        beta=5,
    ),
    **_BC,
)


_sigma_vae = dict(
    ckpt_dir=_EXP_DIR + 'sigma_vae/',
    state_normalizer=dict(
        cls='DictNormalizer',
        observation=dict(cls='RunningMeanStdNormalizer'),
        achieved_goal=dict(cls='PixelRescaleNormalizer', zero_one_range=False),
        desired_goal=dict(which='achieved_goal', update_with=False),  # use same normalizer
    ),
    embedding_head_wrappers=[dict(cls='SiameseGCEmbeddingHeadWrapper')],
    embedding_head=dict(
        cls='SigmaVAE',
        detach_embedding=True,  # do not have agent loss update encoder
        encoder_network_params='plan2explore_encoder',
        decoder_network_params='plan2explore_decoder',
        embedding_dim=_EMBEDDING_DIM,
        optimal_sigma=True,
    ),
    **_BC,
)


_time_nce = dict(
    ckpt_dir=_EXP_DIR + 'time_nce/',
    state_normalizer=dict(
        cls='DictNormalizer',
        observation=dict(cls='RunningMeanStdNormalizer'),
        achieved_goal=dict(cls='PixelRescaleNormalizer', zero_one_range=True),
        desired_goal=dict(which='achieved_goal', update_with=False),  # use same normalizer
    ),
    embedding_head_wrappers=[dict(cls='SiameseGCEmbeddingHeadWrapper')],
    embedding_head=dict(
        cls='NCE',
        encoder_network_params='plan2explore_encoder',
        embedding_dim=_EMBEDDING_DIM,
        pair_type='sampled_pair',  # temporal contrast
        #
        similarity='bilinear',
        #
        # similarity='dotproduct',
        # normalize_projected=True,
        #
        # similarity='negdist',
        # normalize_projected=True,
        #
        projection_head_layers=2,
        detach_embedding=True,  # do not have agent loss update encoder
        aug_params=dict(
            cls='ImageSequential',
            same_on_batch=False,
            aug=[
                # dict(cls='RandomConv'),
                dict(cls='RandomShift', pad=6),
            ],
        ),
        tau=0.01,
        use_target_for_pair=True,
        detach_projected_pair=True,
    ),
    buffer=dict(
        buffer_wrappers=[
            dict(
                cls='CPCPairSamplerBufferWrapper',
                l=3,
                M=1,
            )
        ],  # temporal contrast
    ),
    **_BC,
)


_time_nce_explicit_neg = update_cfg(  # TODO: collapses, not enough negatives?
    _time_nce,
    dict(
        ckpt_dir=_EXP_DIR + 'time_nce_explicit_neg/',
        embedding_head=dict(
            batch_as_neg=False,
            #
            similarity='negdist',
            use_target_for_pair=False,
            detach_projected_pair=False,
        ),
        buffer=dict(
            buffer_wrappers=[
                dict(
                    cls='HTMPairSamplerBufferWrapper',
                    l=3,
                    M=2,
                    num_negatives_per=1,
                )
            ]
        ),
    ),
)


_sim_siam = dict(
    ckpt_dir=_EXP_DIR + 'sim_siam/',
    state_normalizer=dict(
        cls='DictNormalizer',
        observation=dict(cls='RunningMeanStdNormalizer'),
        achieved_goal=dict(cls='PixelRescaleNormalizer', zero_one_range=True),
        desired_goal=dict(which='achieved_goal', update_with=False),  # use same normalizer
    ),
    embedding_head_wrappers=[dict(cls='SiameseGCEmbeddingHeadWrapper')],
    embedding_head=dict(
        cls='SimSiam',
        encoder_network_params='plan2explore_encoder',
        embedding_dim=_EMBEDDING_DIM,
        prediction_head_bottleneck_dim=_EMBEDDING_DIM,  # not bothering with bottleneck
        detach_embedding=True,  # do not have agent loss update encoder
        optim_params=dict(cls='Adam', lr=1e-3, weight_decay=1e-4),
        aug_params=dict(
            cls='ImageSequential',
            same_on_batch=False,
            aug=[
                # dict(cls='RandomConv'),
                dict(cls='RandomShift', pad=6),
            ],
        ),
    ),
    **_BC,
)


_sim_siam_nodetach = update_cfg(
    _sim_siam,
    dict(
        ckpt_dir=_EXP_DIR + 'sim_siam_nodetach/',
        embedding_head=dict(detach_embedding=False),
    ),
)


_sim_siam_origaugs = update_cfg(
    _sim_siam,
    dict(
        ckpt_dir=_EXP_DIR + 'sim_siam_origaugs/',
        embedding_head=dict(
            detach_embedding=True,
            forward_aug_params=dict(
                cls='ImageSequential',
                same_on_batch=True,
                aug=[
                    dict(cls='Resize', size=(224, 224)),
                ],
            ),
            aug_params=dict(
                cls='ImageSequential',
                same_on_batch=False,
                aug=[
                    dict(cls='RandomResizedCrop', p=1.0, size=(224, 224), scale=(0.2, 1.0)),
                    dict(
                        cls='ColorJitter',
                        p=0.8,
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1,
                    ),  # not strengthened
                    dict(cls='RandomGrayscale', p=0.2),
                    #
                    # kernel size is odd and 10% of image height/width, SimCLR Appendix A.
                    # 224 // 10 + 1 == 23
                    # Zero-padding, border_type used to call torch.functional.pad
                    # dict(
                    #     cls='RandomGaussianBlur',
                    #     p=0.2,
                    #     kernel_size=(23, 23),
                    #     sigma=(0.1, 2.0),
                    #     border_type='constant',
                    # ),
                    dict(cls='RandomHorizontalFlip', p=0.5),
                ],
            ),
        ),
        env=dict(frame_size=None),  # randomresizedcropping so don't resize
    ),
)


_sim_siam_origaugs_nodetach = update_cfg(
    _sim_siam_origaugs,
    dict(
        ckpt_dir=_EXP_DIR + 'sim_siam_origaugs_nodetach/',
        embedding_head=dict(detach_embedding=False),
    ),
)


from exp_compare_bc import _bc

_random_embedding = update_cfg(
    _bc,
    dict(
        ckpt_dir=_EXP_DIR + 'random_embedding/',
        embedding_head=dict(
            detach_embedding=True,
        ),
    ),
)


_BASE_ = 'expbase.py'

_VARIANTS_ = [
    _beta_vae,
    _beta_vae_nodetach,
    _rae,
    _sigma_vae,
    _time_nce,
    _impala_encoder,
    _time_nce_explicit_neg,
    _sim_siam,
    _sim_siam_nodetach,
    _sim_siam_origaugs,
    _sim_siam_origaugs_nodetach,
    _random_embedding,
]

from expshared import _SEEDS_
