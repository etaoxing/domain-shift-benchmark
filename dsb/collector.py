from dsb.dependencies import *


class Collector:
    def __init__(
        self,
        policy,
        buffer,
        env,
        initial_collect_steps=0,
        timeout_set_done=True,
        reward_shaper=None,
    ):
        self.buffer = buffer
        self.env = env
        self.policy = policy

        self.steps = 0
        self.initial_collect_steps = initial_collect_steps
        self.timeout_set_done = timeout_set_done
        self.reward_shaper = reward_shaper

        self.reset()

    @property
    def sum_diagnostics_keys(self):
        return [
            '_diagnostics/runtime_policy_act',
            '_diagnostics/runtime_env_step',
            '_diagnostics/runtime_buffer_add',
        ]

    def reset(self):
        self.stats = collections.defaultdict(list)
        self.env_stats = {i: collections.defaultdict(int) for i in range(self.env.num_envs)}
        self.state = self.env.reset()

        self.state_embedding = None
        self.next_state_embedding = None

    def add_to_buffer(self):  # bytransition
        for env_idx, info in enumerate(self.info):
            done = self.done[env_idx]

            if done:
                next_state = info.pop('terminal_observation')
                last_step = True
            else:
                next_state = {k: v[env_idx] for k, v in self.next_state.items()}
                last_step = False

            if done and not self.timeout_set_done:
                done = info.get('success', False)

            state = {k: v[env_idx] for k, v in self.state.items()}

            # NOTE: set goal to be the same within a transition, planning
            next_state['desired_goal'] = state['desired_goal']  # same_goal_in_transition

            action = self.action[env_idx]
            reward = self.reward[env_idx]

            self.buffer.add(
                state, action, next_state, reward, done, last_step=last_step, env_idx=env_idx
            )

            self.env_stats[env_idx]['return'] += reward
            # if reward != 0: self.env_stats[env_idx]['nonzero_return'] += 1
            self.env_stats[env_idx]['length'] += 1
            if last_step:
                # assert self.env_stats[env_idx]['length'] <= self.env.envs[env_idx].duration

                _success = info.pop('success', None)
                if _success is not None:
                    _success = float(_success)
                    # convert so doesn't log like
                    # collected/success/Max                                    False
                    # collected/success/Mean                                0.000000
                    # collected/success/Median                              0.000000
                    # collected/success/Min                                    False
                    # collected/success/Std                                 0.000000

                self.env_stats[env_idx]['success'] = _success
                for k, v in self.env_stats[env_idx].items():
                    self.stats[k].append(v)
                self.env_stats[env_idx].clear()

                info.pop('timeout', None)
                # log other info stats
                for k, v in info.items():
                    if k[0] == '_':  # '_goaldiff_{}'
                        self.stats[f'eoe/{k}'].append(v)
                    else:
                        self.stats[k].append(v)

    def step(self, num_steps):
        for _ in range(num_steps):
            policy_act_start = time.perf_counter()
            if self.steps < self.initial_collect_steps:
                self.action = [self.env.action_space.sample() for _ in range(self.env.num_envs)]
            else:
                self.state, self.action = self.policy.select_action(
                    self.state, state_embedding=self.state_embedding
                )  # replace state since search policy may modify goal

                # if self.steps > 1000: self.env.envs[0].render()
            policy_act_end = time.perf_counter()
            self.stats['_diagnostics/runtime_policy_act'].append(policy_act_end - policy_act_start)

            env_step_start = time.perf_counter()
            self.next_state, self.reward, self.done, self.info = self.env.step(np.copy(self.action))
            env_step_end = time.perf_counter()
            self.stats['_diagnostics/runtime_env_step'].append(env_step_end - env_step_start)

            if self.reward_shaper is not None:
                self.next_state_embedding, self.reward = self.reward_shaper(
                    self.next_state, self.reward, self.done
                )

            add_to_buffer_start = time.perf_counter()
            self.add_to_buffer()
            add_to_buffer_end = time.perf_counter()
            self.stats['_diagnostics/runtime_buffer_add'].append(
                add_to_buffer_end - add_to_buffer_start
            )

            self.steps += 1
            self.state = self.next_state
            self.state_embedding = self.next_state_embedding

        # clear embeddings, otherwise may be stale in the proceeding call to step
        self.state_embedding = None
        self.next_state_embedding = None

        collect_info = dict(self.stats.copy())
        self.stats.clear()
        return collect_info

    @classmethod
    def get_trajectories(
        cls, policy, eval_env, deterministic=True, reset_env=True, initial_state=None
    ):
        num_envs = eval_env.num_envs

        ep_state = [[] for _ in range(num_envs)]
        ep_action = [[] for _ in range(num_envs)]
        ep_reward = [[] for _ in range(num_envs)]
        ep_done = [[] for _ in range(num_envs)]
        ep_env_stats = [{} for _ in range(num_envs)]

        if reset_env:
            _state = eval_env.reset()
        else:
            _state = initial_state

        env_idx_done = set()

        ep_goal = _state['desired_goal']
        while len(env_idx_done) != num_envs:
            _state, _action = policy.select_action(
                _state, deterministic=deterministic, update_stats=False
            )
            waypoint = _state['desired_goal']  # policy changes waypoint, so update

            _next_state, _reward, _done, _info = eval_env.step(np.copy(_action))

            # add to buffers
            for env_idx, info in enumerate(_info):
                if env_idx in env_idx_done:  # skip if had already completed
                    continue

                done = _done[env_idx]

                if done:
                    env_idx_done.add(env_idx)

                    next_state = info.pop('terminal_observation')
                    last_step = True
                else:
                    next_state = {k: v[env_idx] for k, v in _next_state.items()}
                    last_step = False

                state = {k: v[env_idx] for k, v in _state.items()}
                ep_state[env_idx].append(state)

                ep_action[env_idx].append(_action[env_idx])
                ep_reward[env_idx].append(_reward[env_idx])
                ep_done[env_idx].append(done)

                if last_step:
                    ep_state[env_idx].append(next_state)

                    ep_env_stats[env_idx]['return'] = sum(ep_reward[env_idx])
                    ep_env_stats[env_idx]['length'] = len(ep_reward[env_idx])
                    ep_env_stats[env_idx]['success'] = info.pop('success', None)

                    info.pop('timeout', None)
                    # log other info stats
                    for k, v in info.items():
                        if k[0] == '_':
                            ep_env_stats[env_idx][f'eoe/{k}'] = v
                        else:
                            ep_env_stats[env_idx][k] = v

            # update state
            _state = _next_state
        return ep_state, ep_action, ep_reward, ep_done, ep_env_stats

    @classmethod
    def get_trajectory_single_env(
        cls, policy, eval_env, deterministic=True, reset_env=True, initial_state=None
    ):
        ep_states = []
        ep_actions = []
        ep_rewards = []
        ep_dones = []
        ep_infos = []

        if reset_env:
            state = eval_env.reset()
        else:
            state = initial_state

        ep_goal = state['desired_goal']
        while True:
            _state = {k: np.array([v]) for k, v in state.items()}  # batch
            _state, action = policy.select_action(
                _state, deterministic=deterministic, update_stats=False
            )
            # TODO: if with_context in planning policy, then may be overriding train envs
            action = action[0]

            waypoint = _state['desired_goal'][0]
            state['desired_goal'] = waypoint  # policy changes waypoint, so update

            ep_states.append(state)
            ep_actions.append(action)

            state, reward, done, info = eval_env.step(np.copy(action))

            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_infos.append(info)

            if done:
                ep_states.append(info.pop('terminal_observation'))
                break

        return ep_states, ep_actions, ep_rewards, ep_dones, ep_infos
