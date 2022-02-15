import numpy as np
import gym


class DummyKitchenEnv(gym.Env):
    # from https://github.com/google-research/relay-policy-learning/blob/4230de23d7de8081d9f648f8017e2d42fc32e5b2/adept_envs/adept_envs/franka/kitchen_multitask_v0.py#L37
    N_DOF_ROBOT = 9
    N_DOF_OBJECT = 21

    def __init__(self, **kwargs):
        self.params = kwargs

        action_dim = (
            8
            if self.params['ctrl_mode'] == 'mocap' and self.params['rot_use_euler']
            else self.N_DOF_ROBOT
        )
        if self.params.get('binary_gripper', False):
            action_dim -= 1

        act_lower = -1 * np.ones((action_dim,))
        act_upper = 1 * np.ones((action_dim,))
        self.action_space = gym.spaces.Box(act_lower, act_upper)

    def __getattr__(self, attr):
        return self.params[attr]

    def _get_obs_dict(self):
        obs_dict = {
            'robot_qp': np.zeros(self.N_DOF_ROBOT),
            'robot_qv': np.zeros(self.N_DOF_ROBOT),
            'obj_qp': np.zeros(self.N_DOF_OBJECT),
            'obj_qv': np.zeros(self.N_DOF_OBJECT),
        }
        if self.params['with_obs_ee']:
            obs_dict['ee_qp'] = np.zeros(6 if self.params['rot_use_euler'] else 7)
        if self.params['with_obs_forces']:
            obs_dict['ee_forces'] = np.zeros(12)
        return obs_dict
