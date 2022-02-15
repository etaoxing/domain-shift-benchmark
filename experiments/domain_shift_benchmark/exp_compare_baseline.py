from dsb.utils import update_cfg

_EXP_DIR = './workdir_experiments/dsb/exp_compare_baseline/'


_random_action = dict(
    ckpt_dir=_EXP_DIR + 'random_action/',
    runner=dict(num_iterations=100, eval_interval=100),
    agent=dict(cls='NoopAgent'),
    policy=[dict(cls='EpsilonGreedyPolicy', epsilon=1)],
)
_random_action = update_cfg(
    _random_action,
    dict(runner=dict(ckpt_interval=None, eval_func_params=dict(d0=dict(sub_eval_interval=None)))),
)


_demo_playback = dict(
    ckpt_dir=_EXP_DIR + 'demo_playback/',
    runner=dict(num_iterations=100, eval_interval=100),
    agent=dict(cls='NoopAgent'),
    policy=[dict(cls='ContextDemoPolicy'), dict(cls='PlaybackDemoPolicy')],
)
_demo_playback = update_cfg(
    _demo_playback,
    dict(runner=dict(ckpt_interval=None, eval_func_params=dict(d0=dict(sub_eval_interval=None)))),
)


_BASE_ = 'expbase.py'

_VARIANTS_ = [
    _random_action,
    _demo_playback,
]

from expshared import _SEEDS_
