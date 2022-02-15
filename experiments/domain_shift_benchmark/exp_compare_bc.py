from dsb.utils import update_cfg

from expshared import (
    _EMBEDDING_DIM,
    _POLICY_OPTIM,
    _IMITATOR,
    _BC,
    _EMBEDDING_HEAD,
    _GC_STATE_KEYS,
)

_EXP_DIR = './workdir_experiments/dsb/exp_compare_bc/'


_bc = dict(
    ckpt_dir=_EXP_DIR + 'bc/',
    **_EMBEDDING_HEAD,
    **_BC,
)


_bc_mse = dict(
    ckpt_dir=_EXP_DIR + 'bc_mse/',
    **_EMBEDDING_HEAD,
    agent=dict(
        cls='ImitationAgent',
        imitator_params=dict(
            cls='MSEBehavioralCloning',
            actor_params=dict(
                state_keys=('observation', 'achieved_goal'),
                cls='Actor',
                hidden_dim=256,
            ),
            actor_optim_params=_POLICY_OPTIM,
        ),
    ),
    policy=[],
)


_gcbc_her = dict(
    ckpt_dir=_EXP_DIR + 'gcbc_her/',
    **_EMBEDDING_HEAD,
    **_BC,
    buffer=dict(
        buffer_wrappers=[
            dict(
                cls='OnlineHERBufferWrapper',
                k=4,
                # future_p=1.0,
                strategy='future',
                future_horizon=32,  # see latent plans from play κ
                relabel_on_sample=False,
                with_relabel_mask=False,  # just relabel goal for GCBC
                substitute_tensor=True,
            ),
        ],
    ),
)
_gcbc_her = update_cfg(
    _gcbc_her,
    dict(agent=dict(imitator_params=dict(actor_params=dict(state_keys=_GC_STATE_KEYS)))),
)


_gcbc = dict(
    ckpt_dir=_EXP_DIR + 'gcbc/',
    runner=dict(
        batch_size_opt=6,
    ),
    **_EMBEDDING_HEAD,
    **_BC,
    buffer=dict(
        sample_mode='trajectory',
        window_bounds=[20, 40],
        buffer_wrappers=[
            dict(cls='CollateTrajectoryBufferWrapper'),
            dict(cls='OnlineGCBCBufferWrapper'),
            dict(cls='UnCollateTrajectoryBufferWrapper'),
        ],
    ),
)
_gcbc = update_cfg(
    _gcbc,
    dict(agent=dict(imitator_params=dict(actor_params=dict(state_keys=_GC_STATE_KEYS)))),
)


_bc_inverse_dynamics = dict(
    ckpt_dir=_EXP_DIR + 'bc_inverse_dynamics/',
    **_EMBEDDING_HEAD,
    **_BC,
    dynamics_head=dict(
        cls='BasicDynamics',
        inverse_model_params=dict(
            cls='InverseModel',
            state_keys=('observation', 'achieved_goal'),
            concat_next_state=True,
        ),
        optim_params=_POLICY_OPTIM,
    ),
)


_bc_inverse_mixture_dynamics = dict(
    ckpt_dir=_EXP_DIR + 'bc_inverse_mixture_dynamics/',
    **_EMBEDDING_HEAD,
    **_BC,
    dynamics_head=dict(
        cls='BasicDynamics',
        inverse_model_params=dict(
            cls='InverseMixtureModel',
            state_keys=('observation', 'achieved_goal'),
            concat_next_state=True,
            hidden_dim=256,
            mix_dist_params=dict(cls='DiscretizedLogisticMixture', num_bins=256),
            model_params=dict(
                n_mixtures=5,
                const_var=False,
                hidden_dim=256,
            ),
        ),
        optim_params=_POLICY_OPTIM,
    ),
)


_BASE_ = 'expbase.py'

_VARIANTS_ = [
    _bc,
    _bc_inverse_dynamics,
    _gcbc,
    _gcbc_her,
    _bc_inverse_mixture_dynamics,
    _bc_mse,
]

from expshared import _SEEDS_
