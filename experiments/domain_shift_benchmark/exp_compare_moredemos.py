from dsb.utils import update_cfg

_EXP_DIR = './workdir_experiments/dsb/exp_compare_moredemos/'

from expshared import (
    _GC_STATE_KEYS,
)

_MORE_DEMOS = dict(
    env=dict(
        filter_demos=[
            'friday_t-microwave,bottomknob,slide,hinge',
            'friday_t-microwave,bottomknob,switch,slide',
            'friday_t-microwave,bottomknob,topknob,hinge',
            'friday_t-microwave,bottomknob,topknob,slide',
            'friday_t-microwave,kettle,bottomknob,hinge',
            'friday_t-microwave,kettle,bottomknob,slide',
            'friday_t-microwave,kettle,slide,hinge',
            'friday_t-microwave,kettle,switch,slide',  # base task
            'friday_t-microwave,kettle,topknob,hinge',
            'friday_t-microwave,kettle,topknob,switch',
            # 'postcorl_t-microwave,bottomknob,switch,slide',
            # 'postcorl_t-microwave,bottomknob,topknob,switch',
            # 'postcorl_t-microwave,kettle,switch,hinge',
            # 'postcorl_t-microwave,switch,slide,hinge',
            # 'postcorl_t-microwave,topknob,switch,hinge',
        ],
    ),
    runner=dict(
        num_iterations=200000,
        eval_func_params=dict(
            d0=dict(
                sub_eval_interval=200000,
            )
        ),
    ),
)


from exp_compare_bc import _gcbc

_gcbc_moredemos_startmicrowave = update_cfg(
    _gcbc,
    dict(
        ckpt_dir=_EXP_DIR + 'gcbc_moredemos_startmicrowave/',
        **_MORE_DEMOS,
    ),
)


from exp_compare_representation import _beta_vae

_gcbc_beta_vae_moredemos_startmicrowave = update_cfg(
    _gcbc_moredemos_startmicrowave,
    dict(
        ckpt_dir=_EXP_DIR + 'gcbc_beta_vae_moredemos_startmicrowave/',
    ),
)
for k in ['state_normalizer', 'embedding_head_wrappers', 'embedding_head']:
    _gcbc_beta_vae_moredemos_startmicrowave[k] = _beta_vae[k]
_gcbc_beta_vae_moredemos_startmicrowave = update_cfg(
    _gcbc_beta_vae_moredemos_startmicrowave,
    dict(agent=dict(imitator_params=dict(actor_params=dict(state_keys=_GC_STATE_KEYS)))),
)


from exp_compare_bc import _gcbc_her

_gcbc_her_moredemos_startmicrowave = update_cfg(
    _gcbc_her,
    dict(
        ckpt_dir=_EXP_DIR + 'gcbc_her_moredemos_startmicrowave/',
        **_MORE_DEMOS,
    ),
)


_BASE_ = 'expbase.py'

_VARIANTS_ = [
    _gcbc_moredemos_startmicrowave,
    _gcbc_her_moredemos_startmicrowave,
]

from expshared import _SEEDS_
