from dsb.dependencies import *

from dsb.utils import (
    set_global_seed,
    set_env_seed,
    torchify,
    set_device,
    TimestampLogger,
    MDPSpace,
    load_cfg,
    cfg_override_extras,
    load_pretrained,
    get_commit,
    is_repo_clean,
    get_diff_patch,
    update_cfg,
)
import dsb.builder as builder


def init_with_demos(dataset_buffer, agent, update_stats=True):
    dataset_buffer.init_dataloader('train_dataset')
    print(f'num batches in train dataloader: {len(dataset_buffer.dataloader)}')

    if update_stats:

        if agent.state_normalizer is not None and agent.state_normalizer.requires_update:

            def update_stats_with_data(data_iter, update_state_normalizer):
                while True:
                    try:
                        ptrs, batch = next(data_iter)
                        (state, _, _, _, _) = batch
                        x = torchify(state)

                        # NOTE: not bothering w/ terminal state in next_state
                        update_state_normalizer(x)
                    except StopIteration:
                        break

            update_stats_with_data(
                iter(dataset_buffer.dataloader),
                agent.state_normalizer.update_stats,
            )
            print('updated state normalizer using train dataset')


def get_vec_env_cls(num_envs):
    if num_envs == 1:
        from dsb.envs.vec_env import DummyVecEnv

        EnvCls = DummyVecEnv
    else:
        try:
            from dsb.envs.vec_env import ShmemVecEnv

            EnvCls = ShmemVecEnv  # w/ python3.8, we can use mp.shared_memory, which is a little faster than mp.Array
        except ImportError as e:
            print(f'{e}, falling back in SubprocVecEnv')
            from dsb.envs.vec_env import SubprocVecEnv

            EnvCls = SubprocVecEnv
    return EnvCls


def split_egl_devices(num_envs, _env_fn, device_ids):
    c = 0
    _env_fns = []
    for i in range(1, num_envs + 1):
        _env_fns.append(functools.partial(_env_fn, egl_device_id=device_ids[c]))
        # split envs evenly among selected gpus for EG
        if i % (num_envs // len(device_ids)) == 0:
            c += 1
    return _env_fns


def setup_run(args, extras):
    if args.exp == 'test':
        if not args.run:
            raise RuntimeError('specify a run directory for testing')

    # load cfg
    cfg = load_cfg(args.cfg)
    cfg = cfg_override_extras(cfg, extras)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # set seed
    cfg.seed = args.seed
    torch.backends.cudnn.benchmark = True
    set_global_seed(cfg.seed)

    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
    torch.set_num_threads(4)

    # set device
    if args.gpu != -1:
        if args.tpu:
            import torch_xla.core.xla_model as xm

            cfg.device = xm.xla_device(n=args.gpu, devkind='TPU')

            from dsb.utils import set_xla

            set_xla(True)
        else:
            assert torch.cuda.is_available()
            cfg.device = torch.device(args.gpu)
    else:
        cfg.device = 'cpu'

    # create run name
    if args.run:
        run = args.run
    else:
        if args.debug:
            raise NotImplementedError
            import tempfile

            args.debug_tmpfile_dir = tempfile.TemporaryDirectory(
                dir=cfg.ckpt_dir
            )  # hold ref, cleaned later
            run = args.debug_tmpfile_dir.name.split('/')[-1]
        else:
            run = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if len(args.tag) > 0:
        run += f'_{args.tag}'

    # create ckpt dir
    cfg.ckpt_dir = os.path.join(cfg.ckpt_dir, run)
    if os.path.exists(cfg.ckpt_dir):
        if args.exp == 'train' and not args.load_ckpt:
            raise ValueError(f'workdir exists: {cfg.ckpt_dir}')
    else:
        os.makedirs(cfg.ckpt_dir)
        if args.exp == 'train':
            import shutil

            shutil.copyfile(args.cfg, os.path.join(cfg.ckpt_dir, 'cfg.py'))

    cfg.tmp_dir = args.tmp_dir

    # create logger
    sys.stdout = TimestampLogger(log_path=os.path.join(cfg.ckpt_dir, f'log_{args.exp}.txt'))
    sys.stderr = sys.stdout

    print('+' * 10)

    # log machine info, commit, args, cfg
    import subprocess

    machine_log = (
        subprocess.check_output(['python', '-m', 'torch.utils.collect_env']).strip().decode()
    )
    # it's fine if cudnn not found, torch binaries include it, check w/ torch.backends.cudnn.version()
    print(machine_log)

    commit = get_commit()
    print(f'commit: {commit}')
    if commit is not None:
        clean = is_repo_clean()
        print(f'repo clean?: {clean}')
        if not clean:
            diff = get_diff_patch()
            with open(os.path.join(cfg.ckpt_dir, 'diff.patch'), 'w') as f:
                f.write(diff)
    print('-' * 10)
    print(args)
    print(extras)
    print(cfg)
    # exit()
    return cfg


def setup_env(cfg):
    env_load_fn = builder.get_env_load_fn(cfg.env_type)
    env_fn = functools.partial(env_load_fn, cfg.env_name, **cfg.env)

    eval_env_cfg = cfg.env.copy()
    eval_env_cfg = update_cfg(cfg.env, cfg.get('eval_env', {}))
    eval_env_fn = functools.partial(env_load_fn, cfg.env_name, **eval_env_cfg)

    EnvCls = get_vec_env_cls(cfg.num_envs)
    EvalEnvCls = get_vec_env_cls(cfg.num_eval_envs)

    # eval_env uses EGL_DEVICE_ID, and so does train_env if not specified
    if os.environ.get('EGL_DEVICE_IDS', False):
        env_fns = split_egl_devices(cfg.num_envs, env_fn, os.environ['EGL_DEVICE_IDS'].split(','))
    else:
        env_fns = [env_fn] * cfg.num_envs
    eval_env_fns = [eval_env_fn] * cfg.num_eval_envs

    env = EnvCls(env_fns, start_method=args.mp_start_method)
    eval_env = EvalEnvCls(eval_env_fns, start_method=args.mp_start_method)

    set_env_seed(env, cfg.seed + 1)
    set_env_seed(eval_env, cfg.seed + 1 + cfg.num_envs)

    print(f'using vec env cls: {EnvCls}')
    print(f'using eval vec env cls: {EvalEnvCls}')

    _eval_env_obs_space = {
        k: (v.shape, v.dtype) for k, v in eval_env.observation_space.spaces.items()
    }
    print(f'eval_env full obs space: {_eval_env_obs_space}')
    return env, eval_env


def setup_agent(cfg, env):
    obs_space = dict(  # select parts of obs_space used by policy
        observation=env.observation_space['observation'],  # proprioception
        achieved_goal=env.observation_space['achieved_goal'],
        desired_goal=env.observation_space['desired_goal'],
    )
    obs_space = gym.spaces.Dict(obs_space)

    if cfg.get('state_normalizer', False):  # TODO: add state normalizer to embedding head?
        state_normalizer = builder.build_normalizer(cfg.state_normalizer, obs_space)
    else:
        state_normalizer = None

    if cfg.get('embedding_head', False):
        # achieved_goal and desired_goal should be the same goal space
        embedding_head = builder.build_embedding_head(
            cfg.embedding_head,
            obs_space['achieved_goal'],
            embedding_head_wrappers=cfg.get('embedding_head_wrappers', None),
        )
        embedding_dim = embedding_head.embedding_dim
        embedding_keys = ['achieved_goal', 'desired_goal']
        assert len(obs_space['observation'].shape) == 1
    else:
        embedding_head = None
        embedding_dim = None
        embedding_keys = []
        # assert len(obs_space['achieved_goal'].shape) == 1

    if cfg.get('pretrained_embedding_head', False):
        freeze = cfg.pretrained_embedding_head.get('freeze', False)
        state_dict_path = os.path.join(
            cfg.pretrained_embedding_head.ckpt_dir.format(seed=cfg.seed), 'agent.pth'
        )
        state_dict = load_pretrained(
            embedding_head,
            state_dict_path,
            'embedding_head',
            device=cfg.device,
            freeze=freeze,
        )

        if 'state_normalizer' in state_dict.keys():
            del state_dict['state_normalizer']['observation']
            raise NotImplementedError
            state_normalizer.load_state_dict(state_dict['state_normalizer'])

        if freeze:
            # prevent updates to state normalizer
            state_normalizer.requires_update['achieved_goal'] = False
            state_normalizer.requires_update['desired_goal'] = False

    mdp_space = MDPSpace(
        obs_space, env.action_space, embedding_dim=embedding_dim, embedding_keys=embedding_keys
    )
    print(mdp_space)

    if cfg.get('dynamics_head', False):
        dynamics_head = builder.build_dynamics_head(
            cfg.dynamics_head,
            mdp_space,
            embedding_head_param_group={'params': embedding_head.parameters()}
            if embedding_head
            else None,
        )
    else:
        dynamics_head = None

    agent = builder.build_agent(
        cfg.agent,
        mdp_space,
        state_normalizer=state_normalizer,
        embedding_head=embedding_head,
        dynamics_head=dynamics_head,
    )
    print(agent)

    if cfg.device != 'cpu':
        set_device(cfg.device)
        agent = agent.to(cfg.device)

    return agent


def run(args, extras):
    # ---
    # setting up experiment, envs, agent, demo data, and buffer storage
    # ---

    cfg = setup_run(args, extras)
    env, eval_env = setup_env(cfg)
    agent = setup_agent(cfg, env)

    if isinstance(env.action_space, gym.spaces.Discrete):
        raise RuntimeError("This codebase only supports continuous action spaces currently.")

    if cfg.get('reward_shaper', False):
        raise NotImplementedError

        from dsb.reward_shaper import RewardShaper

        cfg.reward_shaper.pop('cls')
        reward_shaper = RewardShaper(agent, **cfg.reward_shaper)
    else:
        reward_shaper = None

    if hasattr(env, 'get_train_test_datasets') and cfg.env.get('demo_dir', False):
        train_dataset, test_dataset = env.get_train_test_datasets()
        datasets = dict(train_dataset=train_dataset, test_dataset=test_dataset)
    else:
        train_dataset, test_dataset = None, None
        datasets = None

    buffer = builder.build_buffer(
        cfg.buffer,
        # obs_space,
        env.observation_space,  # use env obs_space so additional data is stored
        env.action_space,
        tmp_dir=cfg.tmp_dir,
        state_normalizer=agent.state_normalizer,
        max_episode_steps=getattr(env, 'duration'),  # TODO: change attribute naming to match
        datasets=datasets,
        ckpt_dir=cfg.ckpt_dir,
        batch_size_opt=cfg.runner.get('batch_size_opt', None),
    )
    print(f'buffer: {buffer}')
    try:
        import atexit

        atexit.register(buffer.free)
    except:
        print('did not register atexit hook to free buffer memory')

    # create a separate eval policy b/c policy may be stateful (planning policy)
    # also lets us use different noise parameters than for training
    if getattr(cfg, 'eval_policy', False):
        eval_policy_cfg = cfg.eval_policy
    else:
        # just use same policy as train policy
        eval_policy_cfg = cfg.policy
        print('using same eval policy as policy')

    eval_policy = builder.build_policy(
        eval_policy_cfg,
        agent,
        action_space=env.action_space,
        buffer=buffer,
        ckpt_dir=cfg.ckpt_dir,
    )
    if eval_policy == agent:
        print('eval policy: using agent directly')
    else:
        print(f'eval policy: {eval_policy}')

    # ---
    # done setup, now running experiment
    # ---

    if args.load_ckpt:
        if cfg.runner.get('cls', None) != 'train_eval_offline':
            raise NotImplementedError

        ckpt_file = os.path.join(cfg.ckpt_dir, args.load_ckpt)
        print(f'loading ckpt from: {ckpt_file}')
        state_dict = torch.load(ckpt_file, map_location=cfg.device)
        agent.load_state_dict(state_dict)

    if args.exp == 'train':

        from dsb.utils import TabularLogger

        tabular_logger = TabularLogger(os.path.join(cfg.ckpt_dir, 'record.csv'))
        tabular_logger.record_cfg(cfg)
        tabular_logger.record_argparse(args, extras)

        eval_func_params = cfg.runner.pop('eval_func_params', None)
        eval_func = builder.build_eval_func(eval_func_params)

        which_runner = cfg.runner.pop('cls', 'train_eval')

        terminated = False
        _e = None
        try:
            if which_runner == 'train_eval':

                if 'demo' in buffer.keys():
                    init_with_demos(buffer['demo'], agent)

                policy = builder.build_policy(
                    cfg.policy,
                    agent,
                    action_space=env.action_space,
                    buffer=buffer,
                    ckpt_dir=cfg.ckpt_dir,
                )
                if policy == agent:
                    print('policy: using agent directly')
                else:
                    print(f'policy: {policy}')

                from dsb.runner import train_eval

                train_eval(
                    policy,
                    eval_policy,
                    agent,
                    buffer,
                    env,
                    eval_env,
                    eval_func=eval_func,
                    tabular_logger=tabular_logger,
                    ckpt_dir=cfg.ckpt_dir,
                    **cfg.runner,
                )

            elif which_runner == 'train_eval_offline':

                init_with_demos(buffer, agent, update_stats=False if args.load_ckpt else True)

                from dsb.runner import train_eval_offline

                train_eval_offline(
                    eval_policy,
                    agent,
                    buffer,
                    env,
                    eval_env,
                    eval_func=eval_func,
                    tabular_logger=tabular_logger,
                    ckpt_dir=cfg.ckpt_dir,
                    **cfg.runner,
                )

            else:
                raise ValueError

        except (Exception, MemoryError, KeyboardInterrupt, SystemExit) as e:
            terminated = True
            _e = e

            import traceback

            print(traceback.format_exc())
            sys.stdout.flush()

        # https://pytorch.org/docs/stable/notes/faq.html#my-out-of-memory-exception-handler-can-t-allocate-memory
        if terminated:
            torch.save(agent.state_dict(), os.path.join(cfg.ckpt_dir, 'agent_exception.pth'))
            # torch.save(buffer, os.path.join(cfg.ckpt_dir, 'rb_exception.pth'))

            env.close()
            eval_env.close()
            raise _e

        torch.save(agent.state_dict(), os.path.join(cfg.ckpt_dir, 'agent.pth'))
        # torch.save(buffer, os.path.join(cfg.ckpt_dir, 'rb.pth'))

        env.close()
        eval_env.close()

        print(cfg.ckpt_dir)

    elif args.exp == 'test':

        # ckpt_file = os.path.join(cfg.ckpt_dir, 'agent.pth')
        # agent.load_state_dict(torch.load(ckpt_file, map_location='cpu'))
        agent.eval()

        # policy = builder.build_policy(
        #     cfg.policy,
        #     agent,
        #     action_space=env.action_space,
        #     buffer=buffer,
        # )
        # print(f'policy: {policy}')

        env = eval_env.envs[0]

        state = env.reset()
        c = 0

        t = []
        while True:
            # del state['desired_goal']
            _state, action = agent.select_action({k: [v] for k, v in state.items()})
            action = action[0]
            state, reward, done, info = env.step(action)

            image = env.render(mode='rgb_array')
            t.append(image)

            c += 1
            if done:
                break

        from dsb.utils import save_video

        save_video(os.path.join(cfg.ckpt_dir, 'a.avi'), np.array(t), channel_first=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, help='path to config file')
    parser.add_argument('-e', '--exp', type=str, default='test', choices=['train', 'test'])
    parser.add_argument(
        '-r', '--run', type=str, default='', help='name of run ckpt dir to load from'
    )
    parser.add_argument(
        '--load_ckpt', default=None, type=str, help='subpath in ckpt_dir of ckpt.pth to load'
    )
    parser.add_argument('--tag', type=str, default='', help='postfix tag for run')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    parser.add_argument('--tpu', action='store_true', help='google colab xla tpu')
    parser.add_argument(
        '-t', '--tmp_dir', type=str, default='/nvme-scratch', help='large file path for buffer'
    )
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--mp_start_method', type=str, default='forkserver', choices=['forkserver', 'spawn']
    )

    args, extras = parser.parse_known_args()

    # https://github.com/pytorch/pytorch/issues/37377#issuecomment-629610327
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    try:
        torch.multiprocessing.set_start_method(args.mp_start_method)
    except Exception as e:
        raise e
        print(e, torch.multiprocessing.get_start_method())

    np.rng = np.random.default_rng()  # gets set to something process-safe later in set_worker_seed

    run(args, extras)
