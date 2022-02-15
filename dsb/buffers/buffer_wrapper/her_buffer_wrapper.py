from dsb.dependencies import *

from .buffer_wrapper import BufferWrapper

# implementation as https://stable-baselines.readthedocs.io/en/master/modules/her.html
class HERBufferWrapper(BufferWrapper):
    def __init__(
        self,
        buffer,
        k=4,
        strategy='future',
        no_hindsight_on_success=False,
        future_horizon=None,
        reached_goal_func=None,  # offline, so this function should be deterministic
        **kwargs
    ):
        """
        Args:
            k: (int) number of goals to sample per transition
        """
        super().__init__(buffer, **kwargs)

        self.k = k
        self.strategy = strategy
        self.future_horizon = future_horizon
        self.no_hindsight_on_success = no_hindsight_on_success
        self.reached_goal_func = reached_goal_func

        # Buffer for storing transitions of the current episode
        self.episode_transitions = collections.defaultdict(list)

    def set_reached_goal_func(self, reached_goal_func):
        self.reached_goal_func = reached_goal_func

    def add(self, *args, env_idx=0, last_step=False, **kwargs):
        self.episode_transitions[env_idx].append(args)

        if last_step:
            self.store_episode(env_idx)
            self.episode_transitions[env_idx].clear()

    def sample_achieved_goal(self, env_idx, transition_idx, N):
        if self.strategy == 'future':
            # Sample a goal that was observed in the same episode after the current step
            if self.future_horizon is not None:
                b = min(transition_idx + self.future_horizon, N)
            else:
                b = N
            selected_idx = np.rng.integers(transition_idx + 1, b)
        elif self.strategy == 'final':
            # Choose the goal achieved at the end of the episode
            selected_idx = -1
        elif self.strategy == 'episode':
            # Random goal achieved during the episode
            selected_idx = np.rng.integers(N)
        else:
            raise ValueError

        selected_transition = self.episode_transitions[env_idx][selected_idx]
        selected_state = selected_transition[0]
        goal = selected_state['achieved_goal']
        return goal

    def store_episode(self, env_idx):
        transition_idx = 0
        num_transitions = len(self.episode_transitions[env_idx])

        _, _, _, _, success = self.episode_transitions[env_idx][-1]
        while transition_idx < num_transitions:
            transition = self.episode_transitions[env_idx][transition_idx]
            self.buffer.add(
                *transition, last_step=(transition_idx == num_transitions - 1), env_idx=env_idx
            )

            # We cannot sample a goal from the future in the last step of an episode
            if (
                transition_idx == len(self.episode_transitions[env_idx]) - 1
                and self.strategy == 'future'
            ):
                break

            if not (self.no_hindsight_on_success and success):
                sampled_goals = [
                    self.sample_achieved_goal(env_idx, transition_idx, num_transitions)
                    for _ in range(self.k)
                ]
                for sampled_goal in sampled_goals:
                    state, action, next_state, reward, done = copy.deepcopy(transition)

                    state['desired_goal'] = sampled_goal.copy()
                    next_state['desired_goal'] = sampled_goal.copy()
                    reward, done = self.reached_goal_func(next_state)
                    hindsight_transition = (state, action, next_state, reward, done)

                    self.buffer.add_directly(*hindsight_transition)

            transition_idx += 1
