from dsb.dependencies import *
from dsb.buffers.utils import get_timestep_of, sample_pos_pair_indices
from dsb.utils import torchify

from ..sample_buffer_wrapper import SampleBufferWrapper


class OnlineHERBufferWrapper(SampleBufferWrapper):
    def __init__(
        self,
        buffer,
        strategy='future',
        k=4,
        future_p=None,
        future_horizon=None,
        batched_reached_goal_func=None,
        with_relabel_mask=True,
        relabel_context=False,
        relabel_on_sample=True,  # whether to relabel using distance function (deterministic), otherwise relabel in agent if learned distance
        substitute_tensor=False,  # set True w/ demonstrations replay buffer
        **kwargs,
    ):
        super().__init__(buffer, **kwargs)
        assert strategy == 'future'
        self.k = k
        if future_p is not None:
            self.future_p = future_p
        else:
            self.future_p = 1 - (1.0 / (1 + k))
        self.future_horizon = future_horizon
        self.batched_reached_goal_func = batched_reached_goal_func
        self.with_relabel_mask = with_relabel_mask
        self.relabel_context = relabel_context
        self.relabel_on_sample = relabel_on_sample
        self.substitute_tensor = substitute_tensor

        self.init_stats()

    def set_reached_goal_func(self, reached_goal_func):
        self.batched_reached_goal_func = reached_goal_func

    def get_batch(self, batch_it, ptrs):
        ptrs, batch = self.buffer.get_batch(batch_it, ptrs)

        batch_size = len(ptrs)
        batch_indices = np.arange(batch_size)
        her_mask = np.rng.uniform(size=batch_size) < self.future_p
        her_batch_indices = batch_indices[her_mask]
        her_ptrs = ptrs[her_mask]
        her_start_ptrs = self.start_ptr[her_ptrs]

        all_selected_batch_indices = []
        all_selected_episode_ptrs = []
        for s in np.unique(her_start_ptrs):
            # NOTE: using [:] to get numpy array from ShmemNumpyArray
            episode_ptrs = np.array(self.episode_ptrs[s][:])
            N = len(episode_ptrs)
            if N == 1:
                continue  # if t=[0] and N=1, then no future transitions

            start_ptrs_mask = her_start_ptrs == s

            # gets timestep of ptr in episode transitions
            y = her_ptrs[start_ptrs_mask]
            t = get_timestep_of(y, episode_ptrs)

            b = None if self.future_horizon is None else self.future_horizon
            # if finding pair for last transition, then it will be paired with itself
            selected_t = sample_pos_pair_indices(t, N, 1, b)
            selected_episode_ptr = episode_ptrs[selected_t]

            # batch indices for her transition
            selected_batch_indices = her_batch_indices[start_ptrs_mask]
            assert len(selected_batch_indices) == len(selected_episode_ptr)

            # exclude if her selected last transition (don't bother relabeling since no future transitions to select from)
            not_last_transition_mask = t != N - 1
            selected_episode_ptr = selected_episode_ptr[not_last_transition_mask]
            selected_batch_indices = selected_batch_indices[not_last_transition_mask]

            all_selected_batch_indices.append(selected_batch_indices)
            all_selected_episode_ptrs.append(selected_episode_ptr)

        selected_batch_indices = np.concatenate(all_selected_batch_indices)
        selected_episode_ptr = np.concatenate(all_selected_episode_ptrs)

        batch = self.substitute_goal(
            batch, batch_size, selected_batch_indices, selected_episode_ptr
        )
        if self.relabel_on_sample:
            state, action, next_state, reward, done = batch
            reward, done = self.relabel_goal(next_state, reward, done)
            batch = (state, action, next_state, reward, done)
        return ptrs, batch

    def substitute_goal(self, batch, batch_size, selected_batch_indices, relabel_ptrs):
        state, action, next_state, reward, done = batch

        # TODO: should we be getting next_state instead as goal?
        goal = self.buffer.get_state(relabel_ptrs, 'achieved_goal').copy()
        if self.substitute_tensor:
            goal = torchify(goal, dtype=None, device='cpu')

        # don't need to check self.batch_dtype b/c get_batch should already cast
        state['desired_goal'][selected_batch_indices] = goal
        next_state['desired_goal'][selected_batch_indices] = goal

        # TODO: framestack
        if self.relabel_context:
            assert self.buffer.with_context
            # c_goal = self.buffer.get_state(relabel_ptrs, 'c_achieved_goal')
            # state['c_desired_goal'][selected_batch_indices] = c_goal.copy()
            # next_state['c_desired_goal'][selected_batch_indices] = c_goal.copy()

            state['c_desired_goal'][selected_batch_indices] = state['c_achieved_goal'][
                selected_batch_indices
            ]
            next_state['c_desired_goal'][selected_batch_indices] = next_state['c_achieved_goal'][
                selected_batch_indices
            ]

        if '_achieved_task_goal' in self.obs_space.spaces.keys():
            atg = self.buffer.get_state(relabel_ptrs, '_achieved_task_goal').copy()
            if self.substitute_tensor:
                atg = torchify(atg, dtype=None, device='cpu')

            state['_desired_task_goal'][selected_batch_indices] = atg
            next_state['_desired_task_goal'][selected_batch_indices] = atg
        # TODO: replace 'task_goal_id'? should be same since goals are selected within a trajectory

        mask = np.zeros(batch_size, dtype=np.bool)
        mask[selected_batch_indices] = True

        if self.with_relabel_mask:
            next_state['_goal_relabeled_mask'] = mask

        return (state, action, next_state, reward, done)

    def relabel_goal(self, next_state, reward, done, state_embedding=None, inplace=True):
        mask = next_state.pop('_goal_relabeled_mask')
        relabel_next_state = {k: v[mask] for k, v in next_state.items()}

        if state_embedding is not None:
            try:
                state_embedding = {k: v[mask] for k, v in state_embedding.items()}
                r, d = self.batched_reached_goal_func(
                    relabel_next_state, state_embedding=state_embedding
                )
            except TypeError as e:
                raise Exception(f'{e}, try setting relabel_on_sample==True instead')
                r, d = self.batched_reached_goal_func(relabel_next_state)
        else:
            r, d = self.batched_reached_goal_func(relabel_next_state)

        # NOTE: this may not track stats properly when used with dataloader wrapper
        # and relabel_on_sample=True
        self.stats['num_goals_relabeled'] += int(sum(mask))
        self.stats['num_goals_relabeled_reached'] += int(sum(d))

        # reward and done have shape (batch_size, 1)
        if inplace:
            reward[mask], done[mask] = r, d
            return reward, done
        else:
            return r, d

    def init_stats(self):
        self.stats = dict()
        self.stats['num_goals_relabeled'] = 0
        self.stats['num_goals_relabeled_reached'] = 0

    def reset_stats(self):
        self.buffer.reset_stats()
        self.init_stats()

    def get_stats(self):
        s = self.buffer.get_stats()
        stats = self.stats.copy()
        if stats['num_goals_relabeled'] > 0:
            p = stats['num_goals_relabeled_reached'] / float(stats['num_goals_relabeled'])
            stats['perc_goals_relabeled_reached'] = p
        s.update(stats)
        return s
