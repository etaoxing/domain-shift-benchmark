from dsb.dependencies import *

from dsb.collector import Collector
from dsb.utils import set_env_seed, save_video


def compute_domain_shift_stats(v, t):
    _max = np.max(v)
    _mean = np.mean(v)
    _std = np.std(v, ddof=1)

    _rel = _mean / np.mean(t)

    # See http://www.stat.cmu.edu/%7Ehseltman/files/ratio.pdf
    # to estimate RatioStd
    # or could bootstrap, compute pairwise stds and then average

    # TODO: https://arxiv.org/pdf/2105.05249.pdf
    # https://arxiv.org/pdf/1809.03006.pdf
    # better relative metrics

    return _max, _mean, _std, _rel


def eval_kitchenenv_domain_shift(
    eval_id,
    iteration,
    agent,
    eval_env,
    dataset,
    policy=None,
    ckpt_dir='',
    domain_params=[],
    sub_eval_interval=None,
    fix_seed=True,
    **kwargs,
):
    if sub_eval_interval is not None and iteration % sub_eval_interval != 0:
        return {}

    if fix_seed:
        # A fixed seed is used for the eval environment
        # TODO: can also set seed inside Collector.get_trajectories b4 calling env.reset()
        set_env_seed(eval_env, eval_env.base_seed)

    if policy is None:
        policy = agent

    task_stats = {}

    for task_goal_id, task_goal_elements in enumerate(eval_env.task_goals):
        task_goal_tag = eval_env.get_task_goal_tag(task_goal_elements)
        task_ckpt_dir = os.path.join(ckpt_dir, 'domain_shift', str(iteration), task_goal_tag)

        stats_overall = collections.defaultdict(list)
        stats_categorized = {}

        prefix = f'eval-{eval_id}_task-{task_goal_id}-{task_goal_tag}'

        domain_i = 0
        for shift_category, shifts in domain_params.items():
            if shift_category != 'train':
                stats_categorized[shift_category] = collections.defaultdict(list)

            for j, change_env_fns in enumerate(shifts):
                eval_ckpt_dir = os.path.join(task_ckpt_dir, str(domain_i))

                stats = _eval_kitchenenv_change_env_fns(
                    eval_ckpt_dir,
                    change_env_fns,
                    policy,
                    eval_env,
                    reload_model_xml=True,
                    **kwargs,
                )

                for k, v in stats.items():
                    t = f'{prefix}_domainshift-{domain_i}/{k}'
                    task_stats[t] = v

                    if shift_category != 'train':
                        stats_overall[k].append(v)
                        stats_categorized[shift_category][k].append(v)

                domain_i += 1

        # compute average values per category of domain shift
        # also compute the relative score compared to the train domain
        for shift_category, stats in stats_categorized.items():
            for k, v in stats.items():
                # domain_i == 0 is the training domain
                t_0 = f'{prefix}_domainshift-{0}/{k}'
                _max, _mean, _std, _rel = compute_domain_shift_stats(v, task_stats[t_0])

                tm = f'{prefix}_domainshift-{shift_category}/{k}/Mean'
                task_stats[tm] = _mean

                ts = f'{prefix}_domainshift-{shift_category}/{k}/Std'
                task_stats[ts] = _std

                if k.split('/')[-1][0] != '_':
                    tm_rel = f'{prefix}_domainshift-{shift_category}/{k}/RatioMean'
                    task_stats[tm_rel] = _rel

                    tmax = f'{prefix}_domainshift-{shift_category}/{k}/Max'
                    task_stats[tmax] = _max

        # compute average values across all domain_shifts
        # also compute the relative score compared to the train domain
        # TODO: average across categories instead of individual domain shifts?
        for k, v in stats_overall.items():
            # domain_i == 0 is the training domain
            t_0 = f'{prefix}_domainshift-{0}/{k}'
            _max, _mean, _std, _rel = compute_domain_shift_stats(v, task_stats[t_0])

            tm = f'{prefix}_domainshift-all/{k}/Mean'
            task_stats[tm] = _mean

            ts = f'{prefix}_domainshift-all/{k}/Std'
            task_stats[ts] = _std

            if k.split('/')[-1][0] != '_':
                tm_rel = f'{prefix}_domainshift-all/{k}/RatioMean'
                task_stats[tm_rel] = _rel

                tmax = f'{prefix}_domainshift-all/{k}/Max'
                task_stats[tmax] = _max

    return task_stats


def eval_kitchenenv_generalize_single_object(
    eval_id,
    iteration,
    agent,
    eval_env,
    dataset,
    policy=None,
    ckpt_dir='',
    eval_unchanged=True,
    eval_related=True,
    eval_unrelated=True,
    fix_seed=True,
    **kwargs,
):
    from kitchen_shift.constants import OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS

    if fix_seed:
        # A fixed seed is used for the eval environment
        # TODO: can also set seed inside Collector.get_trajectories b4 calling env.reset()
        set_env_seed(eval_env, eval_env.base_seed)

    # for each task
    # we have two types of generalization
    # a. task object different starting state
    # b. env object (unrelated to task) different starting state
    # for now, we only manipulate a single object

    if policy is None:
        policy = agent

    task_stats = {}
    for task_goal_id, task_goal_elements in enumerate(eval_env.task_goals):
        task_goal_tag = eval_env.get_task_goal_tag(task_goal_elements)

        tge_a = set(task_goal_elements)
        tge_b = set(OBS_ELEMENT_GOALS.keys()).difference(tge_a)

        task_ckpt_dir = os.path.join(
            ckpt_dir, 'generalize_single_object', str(iteration), task_goal_tag
        )

        prefix = f'eval-{eval_id}_task-{task_goal_id}-{task_goal_tag}'

        if eval_unchanged:
            # original starting state, no objects changed
            objects_to_change = []
            change_env_fns = [
                ('set_task_goal_id', task_goal_id),
                # ('change_objects_done', objects_to_change),
            ]
            eval_ckpt_dir = os.path.join(
                task_ckpt_dir, 'changed_' + ','.join(sorted(objects_to_change))
            )
            stats_c = _eval_kitchenenv_change_env_fns(
                eval_ckpt_dir, change_env_fns, policy, eval_env, **kwargs
            )
            for k, v in stats_c.items():
                task_stats[f'{prefix}_obj-unchanged/{k}'] = v

        if eval_related:
            # change objects related to task
            stats_a = collections.defaultdict(list)
            for o in tge_a:
                objects_to_change = [o]
                change_env_fns = [
                    ('set_task_goal_id', task_goal_id),
                    ('change_objects_done', objects_to_change),
                ]
                eval_ckpt_dir = os.path.join(
                    task_ckpt_dir, 'changed_' + ','.join(sorted(objects_to_change))
                )
                o_trajectory_stats = _eval_kitchenenv_change_env_fns(
                    eval_ckpt_dir, change_env_fns, policy, eval_env, **kwargs
                )
                for k, v in o_trajectory_stats.items():
                    stats_a[k] += v  # concatenate
            for k, v in stats_a.items():
                task_stats[f'{prefix}_obj-related/{k}'] = v

        if eval_unrelated:
            # change objects unrelated to task
            stats_b = collections.defaultdict(list)
            for o in tge_b:
                objects_to_change = [o]
                change_env_fns = [
                    ('set_task_goal_id', task_goal_id),
                    ('change_objects_done', objects_to_change),
                ]
                eval_ckpt_dir = os.path.join(
                    task_ckpt_dir, 'changed_' + ','.join(sorted(objects_to_change))
                )
                o_trajectory_stats = _eval_kitchenenv_change_env_fns(
                    eval_ckpt_dir, change_env_fns, policy, eval_env, **kwargs
                )
                for k, v in o_trajectory_stats.items():
                    stats_b[k] += v  # concatenate
            for k, v in stats_b.items():
                task_stats[f'{prefix}_obj-unrelated/{k}'] = v

    return task_stats


def _eval_kitchenenv_change_env_fns(
    eval_ckpt_dir,
    change_env_fns,
    policy,
    eval_env,
    reload_model_xml=False,
    deterministic_policy=True,
    num_evals_per_task=10,
    save_video_interval=1,
    render_key='achieved_goal',
):
    stats = collections.defaultdict(list)

    eval_env.env_method('reset_domain_changes')
    for change_env_fn in change_env_fns:
        eval_env.env_method(*change_env_fn)
    eval_env.reset(reload_model_xml=reload_model_xml)

    for i in range(num_evals_per_task):
        ep_state, ep_action, ep_reward, ep_done, ep_env_stats = Collector.get_trajectories(
            policy,
            eval_env,
            deterministic=deterministic_policy,
        )
        # ep_state[-1] should be ep_info[-1]['terminal_observation]

        for env_idx in range(eval_env.num_envs):
            if ((i * eval_env.num_envs) + env_idx) % save_video_interval == 0:
                if render_key == 'achieved_goal':
                    # save video
                    ep_frames = np.stack([x['achieved_goal'] for x in ep_state[env_idx]])
                    ep_goals = np.stack([x['desired_goal'] for x in ep_state[env_idx]])
                    video = np.concatenate([ep_frames, ep_goals], axis=-1)
                else:
                    video = np.stack([x[render_key] for x in ep_state[env_idx]])

                # check if grayscale and convert to RGB
                if video.shape[1] == 1:
                    video = video.repeat(3, axis=1)

                save_video(
                    os.path.join(eval_ckpt_dir, f'{i}_env{env_idx}.avi'), video, channel_first=True
                )

            env_stats = ep_env_stats[env_idx]
            # for k, v in env_stats.items():
            # for k in ['return', 'length', 'success', 'success_num_obj']:
            for k in ['success_num_obj']:
                v = env_stats[k]
                stats[k].append(v)

            for k, v in env_stats.items():
                if k.startswith('eoe/_goaldiff'):
                    stats[k].append(v)

    # clear changes
    eval_env.env_method('reset_domain_changes')
    if reload_model_xml:  # only need to reload if modified xml
        eval_env.reset(reload_model_xml=True)
    return stats
