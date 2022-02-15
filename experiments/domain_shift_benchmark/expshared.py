_EMBEDDING_DIM = 128

_POLICY_OPTIM = dict(cls='Adam', lr=1e-3, eps=0.01)

_IMITATOR = dict(
    cls='BehavioralCloning',
    mix_dist_params=dict(cls='DiscretizedLogisticMixture', num_bins=256),
    actor_params=dict(
        cls='MixtureActor',
        state_keys=('observation', 'achieved_goal'),
        n_mixtures=5,
        const_var=False,
        hidden_dim=256,
    ),
    actor_optim_params=_POLICY_OPTIM,
)

_BC = dict(
    agent=dict(
        cls='ImitationAgent',
        imitator_params=_IMITATOR,
    ),
    policy=[],
)

_EMBEDDING_HEAD = dict(
    state_normalizer=dict(
        cls='DictNormalizer',
        observation=dict(cls='RunningMeanStdNormalizer'),
        achieved_goal=dict(cls='PixelRescaleNormalizer', zero_one_range=True),
        desired_goal=dict(which='achieved_goal', update_with=False),  # use same normalizer
    ),
    embedding_head_wrappers=[dict(cls='SiameseGCEmbeddingHeadWrapper')],
    embedding_head=dict(
        cls='BasicCNN',
        encoder_network_params='plan2explore_encoder',
        embedding_dim=_EMBEDDING_DIM,
        output_activation_params=None,
    ),
)

_GC_STATE_KEYS = ('observation', 'achieved_goal', 'desired_goal')

_SEEDS_ = [907140376, 1052698772, 895114834, 590507093]
