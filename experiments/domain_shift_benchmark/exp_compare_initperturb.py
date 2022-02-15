from dsb.utils import update_cfg

_EXP_DIR = './workdir_experiments/dsb/exp_compare_initperturb/'

_STATEVEC_RENDER = dict(
    env=dict(
        use_pixels=False,
    ),
    eval_env=dict(
        separate_render=True,
    ),
    runner=dict(
        eval_func_params=dict(
            s0=dict(render_key='render'),
            d0=dict(render_key='render'),
        )
    ),
)

_EVAL_ORIG = dict(
    eval_env=dict(
        kitchen_shift_params=dict(
            init_random_steps_set=None,
            init_perturb_robot_ratio=None,
            init_perturb_object_ratio=None,
            rng_type='legacy',
        ),
    )
)


from expshared import _BC

_bc_statevec = update_cfg(
    _BC,
    dict(
        ckpt_dir=_EXP_DIR + 'bc_statevec/',
        **_STATEVEC_RENDER,
    ),
)


_bc_statevec_evalorig = update_cfg(
    _bc_statevec,
    dict(
        ckpt_dir=_EXP_DIR + 'bc_statevec_evalorig/',
        **_EVAL_ORIG,
    ),
)


from exp_compare_bc import _bc

_bc_evalorig = update_cfg(
    _bc,
    dict(
        ckpt_dir=_EXP_DIR + 'bc_evalorig/',
        **_EVAL_ORIG,
    ),
)


from exp_compare_representation import _beta_vae

_beta_vae_evalorig = update_cfg(
    _beta_vae,
    dict(
        ckpt_dir=_EXP_DIR + 'bc_beta_vae_evalorig/',
        **_EVAL_ORIG,
    ),
)


from exp_compare_baseline import _demo_playback

_demo_playback_evalorig = update_cfg(
    _demo_playback,
    dict(
        ckpt_dir=_EXP_DIR + 'demo_playback_evalorig/',
        **_EVAL_ORIG,
    ),
)


_BASE_ = 'expbase.py'

_VARIANTS_ = [
    _bc_statevec,
    _bc_statevec_evalorig,
    _bc_evalorig,
    _beta_vae_evalorig,
    _demo_playback_evalorig,
]

from expshared import _SEEDS_
