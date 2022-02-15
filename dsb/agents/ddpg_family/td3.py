from .uvf_ddpg import UVFDDPG


class TD3(UVFDDPG):
    def __init__(
        self,
        mdp_space,
        target_policy_smoothing=True,
        actor_optimize_interval=2,
        targets_update_interval=2,
        critic_params=dict(cls='Critic', td3_style=True),
        #
        critic_ensemble_size=2,
        critic_aggregate='min',
        **kwargs,
    ):
        super().__init__(
            mdp_space,
            target_policy_smoothing=target_policy_smoothing,
            actor_optimize_interval=actor_optimize_interval,
            targets_update_interval=targets_update_interval,
            critic_params=critic_params,
            #
            critic_ensemble_size=critic_ensemble_size,
            critic_aggregate=critic_aggregate,
            **kwargs,
        )
        # NOTE: slightly different than official TD3 implementation since we aggregate across
        # the critic ensemble when getting qvalues to compute actor loss, rather than just using
        # a single critic. see https://github.com/sfujim/TD3/blob/7d5030587011a8bb285f457a75068d033e1365d0/TD3.py#L140
