from dsb.dependencies import *
from dsb.utils import (
    TabularLogger,
    torchify,
    untorchify,
    load_image,
    save_image,
    memory_usage_report,
    get_process_gpu_usage,
    copy_cfg,
)

from dsb.collector import Collector


SUM_DIAGNOSTICS_KEYS = [
    '_diagnostics/runtime_policy_act',
    '_diagnostics/runtime_env_step',
    '_diagnostics/runtime_buffer_add',
    '_diagnostics/runtime_agent_opt',
    '_diagnostics/runtime_total_sec',
]


def train_eval(
    policy,
    eval_policy,
    agent,
    buffer,
    env,
    eval_env,
    #
    reward_shaper=None,
    initial_collect_steps=1000,
    timeout_set_done=False,
    #
    num_iterations=int(1e6),
    collect_steps=1,
    delay_collect_by=0,
    opt_steps=1,
    batch_size_opt=64,
    #
    tabular_logger=None,
    eval_func=None,
    log_interval=100,
    eval_interval=10000,
    delay_eval_by=0,
    ckpt_interval=None,
    ckpt_dir='',
    **unused,
):
    assert eval_interval % log_interval == 0
    if ckpt_interval is not None:
        os.makedirs(os.path.join(ckpt_dir, 'ckpt'), exist_ok=True)

    collector = Collector(
        policy,
        buffer,
        env,
        initial_collect_steps=initial_collect_steps,
        timeout_set_done=timeout_set_done,
        reward_shaper=reward_shaper,
    )

    # collector.step(collector.initial_collect_steps)
    # TODO: randomize num steps when multiple envs? https://github.com/astooke/rlpyt/blob/35af87255465b3644747294f7fd1ff6045dab910/rlpyt/samplers/collectors.py#L100

    # NOTE: if we reset, then may be in the middle of an episode, which is
    # cached in buffer.episode_transitions. will result in episodes being
    # concatenated w/ each other. might fix this by only resetting the environment
    # on timeout instead of also success for these initial random steps
    #
    # collector.reset()

    num_episodes = 0
    collect_info_window = collections.defaultdict(list)

    PID = os.getpid()
    i = 0
    # TODO: load iterations
    # >>> df['_diagnostics/total_iterations']
    # 0    100100.0

    start_time = time.perf_counter()
    while i < num_iterations:
        runtime_start = time.perf_counter()

        if i >= delay_collect_by:
            collect_info = collector.step(collect_steps)
            for k, v in collect_info.items():
                if k[0] == '_':
                    collect_info_window[k] += v  # concatenate
                else:
                    collect_info_window[f'collected/{k}'] += v  # concatenate
            num_episodes += len(collect_info.get('length', []))

        agent_opt_start = time.perf_counter()
        if hasattr(buffer, 'ready_batches'):
            buffer.ready_batches(opt_steps, batch_size_opt)
        agent.train()
        opt_info = agent.optimize_for(buffer, iterations=opt_steps, batch_size=batch_size_opt)
        # opt_info = dict(iteration=[i] * opt_steps)
        i += opt_steps
        if hasattr(buffer, 'cleanup_batches'):
            buffer.cleanup_batches()
        agent_opt_end = time.perf_counter()
        collect_info_window['_diagnostics/runtime_agent_opt'].append(
            agent_opt_end - agent_opt_start
        )

        if ckpt_interval is not None and i % ckpt_interval == 0 and i >= delay_eval_by:
            torch.save(agent.state_dict(), os.path.join(ckpt_dir, 'ckpt', f'agent_{i}.pth'))

        if eval_func is not None and i % eval_interval == 0 and i >= delay_eval_by:
            print(f'evaluating iteration = {i}')
            eval_start_time = time.perf_counter()
            agent.eval()
            eval_info = eval_func(
                i,
                agent,
                eval_env,
                buffer.datasets,
                policy=eval_policy,  # using eval_policy instead of policy
                ckpt_dir=ckpt_dir,
            )
            eval_time = time.perf_counter() - eval_start_time
            eval_info['_diagnostics/runtime_eval'] = eval_time
            # print(eval_info) # tabular_logger should print this out
            print('-' * 10)
        else:
            eval_info = {}

        runtime_end = time.perf_counter()
        cml_time = runtime_end - start_time
        opt_info['_diagnostics/elapsed_time_sec'] = cml_time
        collect_info_window['_diagnostics/runtime_total_sec'].append(runtime_end - runtime_start)

        if i % log_interval == 0:
            for k in SUM_DIAGNOSTICS_KEYS:
                if k in collect_info_window.keys():
                    collect_info_window[k] = sum(collect_info_window[k])

            if i - opt_steps >= delay_collect_by:
                collect_info_window['_diagnostics/total_episodes'] = num_episodes
                collect_info_window['_diagnostics/total_collect_steps'] = (
                    collector.steps * collector.env.num_envs
                )
                collect_info_window['_diagnostics/total_replay_ratio'] = (
                    (i - delay_collect_by) * batch_size_opt
                ) / collect_info_window['_diagnostics/total_collect_steps']

                mem_info = memory_usage_report(PID)
                for k, v in mem_info.items():
                    collect_info_window[f'_diagnostics/mem_{k}'] = v

            info = {
                **collect_info_window,
                **opt_info,
                **eval_info,
            }

            if i - opt_steps >= delay_collect_by:
                # TODO: fix stats so it goes thru each policy / buffer wrappers
                policy_stats = policy.get_stats()
                policy.reset_stats()
                policy_stats = {f'_policy_stats/{k}': v for k, v in policy_stats.items()}

                buffer_stats = buffer.get_stats()
                buffer.reset_stats()
                buffer_stats = {f'_buffer_stats/{k}': v for k, v in buffer_stats.items()}

                info.update(policy_stats)
                info.update(buffer_stats)

            tabular_logger.record(info)
            tabular_logger.log(i)
            collect_info_window = collections.defaultdict(list)

            # exit()

            # for k, v in buffer.filebacked.items():
            #     print(v.info)

            # from dsb.utils import gc_obj_report
            # gc_obj_report()
        else:
            tabular_logger.record(opt_info)
