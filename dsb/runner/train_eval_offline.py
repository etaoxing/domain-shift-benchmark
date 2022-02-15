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


def train_eval_offline(
    eval_policy,
    agent,
    buffer,
    env,
    eval_env,
    #
    num_iterations=int(1e6),
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

    PID = os.getpid()
    i = agent.optimize_iterations  # 0 if new run, otherwise loading checkpoint
    start_time = time.perf_counter()

    entered_once = False
    while i < num_iterations or not entered_once:
        if i < num_iterations:
            agent.train()
            opt_info = agent.optimize_for(buffer, iterations=opt_steps, batch_size=batch_size_opt)
            i += opt_steps

            if ckpt_interval is not None and i % ckpt_interval == 0 and i >= delay_eval_by:
                torch.save(agent.state_dict(), os.path.join(ckpt_dir, 'ckpt', f'agent_{i}.pth'))
        else:
            opt_info = dict(iteration=[i])

        if eval_func is not None and i % eval_interval == 0 and i >= delay_eval_by:
            # TODO: replace `with eval_mode(agent)`:
            print(f'evaluating iteration = {i}')
            eval_start_time = time.perf_counter()
            agent.eval()
            eval_info = eval_func(
                i, agent, eval_env, buffer.datasets, policy=eval_policy, ckpt_dir=ckpt_dir
            )
            eval_time = time.perf_counter() - eval_start_time
            eval_info['_diagnostics/runtime_eval'] = eval_time
            # print(eval_info)  # tabular_logger should print this out
            print('-' * 10)
        else:
            eval_info = {}

        cml_time = time.perf_counter() - start_time
        opt_info['_diagnostics/elapsed_time_sec'] = cml_time

        if i % log_interval == 0:
            # if True:
            opt_info['_diagnostics/epoch'] = buffer.epoch
            gpu_usage = get_process_gpu_usage(PID)
            if gpu_usage is not None:
                opt_info['_diagnostics/main_process_gpu_usage'] = gpu_usage

            info = {**opt_info, **eval_info}
            tabular_logger.record(info)

            tabular_logger.log(i)
        else:
            tabular_logger.record(opt_info)

        entered_once = True  # for loading checkpoint and running final eval again
