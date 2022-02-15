from dsb.dependencies import *

from .dummy_kitchen_env import DummyKitchenEnv
from .gc_kitchen_env_wrapper import GoalConditionedKitchenEnvWrapper


def env_load_fn(
    env_name,
    use_dummy_env=False,
    kitchen_shift_params={},
    egl_device_id=None,
    gym_env_wrappers=[],
    worldgen_wrapper=False,
    zero_reward_on_success=False,
    max_episode_steps=280,
    reset_on='either',
    auto_reset=False,
    disable_rewards=False,
    **kwargs,
):
    # assert env_name == 'kitchen_relax-v1'
    # assert env_name == 'kitchen-v0'
    assert env_name == 'kitchen-v1'

    if use_dummy_env:  # use offline demo data w/o mujoco
        env = DummyKitchenEnv(**kitchen_shift_params)
    else:

        try:
            if egl_device_id is not None:
                os.environ['EGL_DEVICE_ID'] = f'{egl_device_id}'

                # import dm_control._render
                # print(egl_device_id, dm_control._render.pyopengl.egl_renderer.EGL_DISPLAY.address)

            import dm_control
            import dm_control.mujoco

            dm_control.mujoco.wrapper.core._maybe_register_license()
            assert dm_control.mujoco.wrapper.core._REGISTERED

            if torch.cuda.is_available():
                # use egl so can specify gpu, https://github.com/deepmind/dm_control/issues/118
                import dm_control._render

                assert dm_control._render.BACKEND == 'egl'

            # import adept_envs
            import kitchen_shift
        except (ImportError, Exception) as e:
            print(e)
            raise e

        env = gym.make(env_name, **kitchen_shift_params)
        env = env.env  # remove gym's TimeLimit wrapper

    env = GoalConditionedKitchenEnvWrapper(env, **kwargs)
    if not use_dummy_env:
        env.generate_task_goals()

    if worldgen_wrapper:
        from kitchen_shift import MujocoWorldgenKitchenEnvWrapper

        env = MujocoWorldgenKitchenEnvWrapper(env)

    for wrapper in gym_env_wrappers:
        env = wrapper(env)

    from dsb.envs.wrappers import TimeLimitWrapper, SparseRewardWrapper

    if zero_reward_on_success:
        env = SparseRewardWrapper(env, zero_reward_on_success=True)

    if max_episode_steps is not None:
        env = TimeLimitWrapper(
            env,
            max_episode_steps=max_episode_steps,
            reset_on=reset_on,
            auto_reset=auto_reset,
            disable_rewards=disable_rewards,
        )

    return env
