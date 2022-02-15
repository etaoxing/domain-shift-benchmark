from dsb.dependencies import *

from .base_policy import BasePolicy
from ..buffers.utils import sample_context_indices


class ContextDemoPolicy(BasePolicy):
    # pick a demonstration to use as context for a trajectory
    def __init__(
        self,
        agent,
        buffer=None,
        add_demo_to_state=False,
        use_same_demo=False,
        T_context=None,
        **kwargs,
    ):
        super().__init__(agent, **kwargs)
        self.context_demo_cached = {}
        self.context_demo_hidden_cached = {}
        self.add_demo_to_state = add_demo_to_state
        self.use_same_demo = use_same_demo
        self.T_context = T_context

        self.demo_dataset = buffer.datasets['train_dataset']

        self.demo_ids_by_task_goal_tag = collections.defaultdict(list)
        for i, demo in enumerate(self.demo_dataset.demos):
            self.demo_ids_by_task_goal_tag[demo['task_goal_tag']].append(i)

    def get_num_demos_of(self, task_goal_id):
        task_goal_tag = self.demo_dataset.task_goal_tags[task_goal_id]
        demo_ids = self.demo_ids_by_task_goal_tag[task_goal_tag]
        return len(demo_ids)

    def get_demo_of(self, task_goal_id, task_demo_idx):
        task_goal_tag = self.demo_dataset.task_goal_tags[task_goal_id]
        # demo_task = self.demo_dataset.task_goals_demo[task_goal_tag]

        demo_ids = self.demo_ids_by_task_goal_tag[task_goal_tag]
        demo_id = demo_ids[task_demo_idx]
        return self.demo_dataset.demos[demo_id]

    def select_action(self, state, deterministic=False, **kwargs):
        self._update_context_demo(state, deterministic=deterministic)

        s = {k: v for k, v in state.items()}

        if self.add_demo_to_state:
            assert self.use_same_demo

            # state['context_demo'] = self.context_demo_cached[env_idx]['obs']['achieved_goal']

            num_envs = len(state['_time_step'])

            context = self.context_demo_cached[0]['obs']['achieved_goal']

            if self.T_context is not None:
                # TODO: deterministic=True cache indices?
                selected_t = sample_context_indices(
                    context.shape[0], self.T_context, sample_sides=True
                )
                context = context[selected_t, ...]

            # Hack
            s['context_demo'] = context[np.newaxis].repeat(num_envs, axis=0)
            s['achieved_goal'] = s['achieved_goal'][:, np.newaxis, ...].copy()

        s, action = self.agent.select_action(s, deterministic=deterministic, **kwargs)

        # if 'context_demo_hidden' in state.keys():
        #     for env_idx in range(num_envs):
        #         t = state['_time_step'][env_idx]
        #         if t == 0:
        #             h = state['context_demo_hidden'][env_idx, ...]
        #             self.context_demo_hidden_cached[env_idx] = h

        # if self.add_demo_to_state:
        #     state.pop('context_demo')

        return state, action

    def _update_context_demo(self, state, deterministic=False):
        num_envs = len(state['_time_step'])

        for env_idx in range(num_envs):
            t = state['_time_step'][env_idx]
            if t == 0:  # episode start
                # pick a new demo
                task_goal_id = int(state['_task_goal_id'][env_idx])
                if deterministic or self.use_same_demo:
                    # just grab the first demo
                    task_demo_idx = 0
                else:
                    # pick random demo
                    num_demos = self.get_num_demos_of(task_goal_id)
                    task_demo_idx = np.rng.integers(0, num_demos)

                selected_demo = self.get_demo_of(task_goal_id, task_demo_idx)
                self.context_demo_cached[env_idx] = selected_demo
