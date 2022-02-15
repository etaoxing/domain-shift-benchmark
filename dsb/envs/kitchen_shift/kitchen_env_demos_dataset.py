import numpy as np
from torch.utils.data import Dataset


class KitchenEnvDemosDataset(Dataset):
    def __init__(
        self,
        demos,
        with_task_goal=False,
        task_goal_tags=None,
        task_goals_demo=None,
        framestack_keys=None,
        framestack_timesteps=None,
    ):
        self.demos = demos

        self.with_task_goal = with_task_goal
        self.task_goal_tags = task_goal_tags
        self.task_goals_demo = task_goals_demo

        self.demo_task_goal_tags = set([x['task_goal_tag'] for x in demos])

        self.generate_transitions()
        self.num_demos = len(self.demos)

        self.set_framestack_params(framestack_keys, framestack_timesteps)

    def set_framestack_params(self, framestack_keys, framestack_timesteps):
        self.framestack_keys = framestack_keys
        self.framestack_timesteps = framestack_timesteps

    def __len__(self):
        return self.num_transitions

    def generate_transitions(self):
        self.num_transitions = sum([demo['num_timesteps'] for demo in self.demos])

        # NOTE: if have a lot of demos then may need to increase precision to int64
        self.transition_demo_idx = np.array(
            [i for i, demo in enumerate(self.demos) for j in range(demo['num_timesteps'])],
            dtype=np.int32,
        )
        self.transition_timestep_idx = np.array(
            [j for i, demo in enumerate(self.demos) for j in range(demo['num_timesteps'])],
            dtype=np.int32,
        )

    def _get_timestep(self, demo, timestep_idx):
        state = {k: v[timestep_idx].copy() for k, v in demo['obs'].items()}
        next_state = {k: v[timestep_idx + 1].copy() for k, v in demo['obs'].items()}

        # set desired goal to last frame of demonstration, [np.newaxis].repeat(N, axis=0)
        dg = demo['obs']['achieved_goal'][-1]
        state['desired_goal'] = dg.copy()
        next_state['desired_goal'] = dg.copy()

        if self.with_task_goal:
            state['_achieved_task_goal'] = demo['obj_state']

        if self.framestack_timesteps is not None:
            fs_t = self.framestack_timesteps.copy()
            fs_indices = (timestep_idx + fs_t).clip(0, demo['num_timesteps'] - 1)
            for k in self.framestack_keys:
                if k == 'desired_goal':
                    v = dg[np.newaxis].repeat(len(fs_t), axis=0)
                else:
                    v = demo['obs'][k][fs_indices].copy()
                state[f'fs_{k}'] = v

            # if self.with_task_goal:
            #     raise NotImplementedError

        action = demo['action'][timestep_idx].copy()
        reward = demo['reward'][timestep_idx].copy()
        done = demo['done'][timestep_idx].copy()

        state['_time_step'] = np.array([timestep_idx])
        next_state['_time_step'] = np.array([timestep_idx + 1])
        # state['_data_path'] = demo['data_path']

        transitions = (state, action, next_state, reward, done)
        return transitions

    def __getitem__(self, idx):
        demo_idx = self.transition_demo_idx[idx]
        timestep_idx = self.transition_timestep_idx[idx]

        demo = self.demos[demo_idx]
        transition = self._get_timestep(demo, timestep_idx)
        return idx, transition
