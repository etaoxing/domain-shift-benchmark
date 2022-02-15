from dsb.dependencies import *


def registry(module_list):
    return {m.__name__: m for m in module_list}


def add_to_registry(r, module_list):
    r.update(registry(module_list))


def parse_params(params):
    assert 'cls' in params.keys()
    params = params.copy()
    c = params.pop('cls')
    return c, params


def build_module(module_registry, module_params, *args, **kwargs):
    c, module_params = parse_params(module_params)
    if isinstance(module_registry, dict):
        ModuleCls = module_registry[c]
    else:
        ModuleCls = getattr(module_registry, c)
    return ModuleCls(*args, **module_params, **kwargs)


def build_optim(optim_params, params=None):
    import torch.optim

    c, optim_params = parse_params(optim_params)

    if c == 'Ranger21':
        from ranger21 import Ranger21

        OptimCls = Ranger21
    else:
        OptimCls = getattr(torch.optim, c)

    o = OptimCls(params, **optim_params)

    # xla patch
    from dsb.utils import XLA

    if XLA:
        import torch_xla.core.xla_model as xm

        def xla_step(o):
            return xm.optimizer_step(o, barrier=True)

        o.step = xla_step
    return o


def build_aug(augmentation_params, img_size=None):
    import dsb.augmentation
    import kornia.augmentation

    if augmentation_params is None:
        return None

    c, container_params = parse_params(augmentation_params)
    ContainerCls = getattr(kornia.augmentation, c)

    aug_list = container_params.pop('aug')

    aug = []
    for a_params in aug_list:
        c, a_params = parse_params(a_params)

        # TODO: inspect if asks for kwarg
        try:
            AugCls = getattr(dsb.augmentation, c)
            a = AugCls(**a_params, img_size=img_size)
        except AttributeError:
            if c == 'Resize':
                AugCls = getattr(kornia.geometry.transform, c)
            else:
                AugCls = getattr(kornia.augmentation, c)
            a = AugCls(**a_params)

        aug.append(a)

    # aug = nn.Sequential(*augs)
    aug = ContainerCls(*aug, **container_params)
    return aug


def build_network_modules(network_modules_params, in_channels=None, out_channels=None):
    import dsb.network_configs

    if isinstance(network_modules_params, str):
        # lookup config by tag
        return build_network_modules(
            getattr(dsb.network_configs, network_modules_params),
            in_channels=in_channels,
            out_channels=out_channels,
        )
    elif isinstance(network_modules_params, list):
        modules = []
        prev_in_channels = in_channels
        for i, module_params in enumerate(network_modules_params):
            c, module_params = parse_params(module_params)
            if 'Conv' in c:
                if module_params.get('in_channels', False):
                    if prev_in_channels is not None:
                        assert prev_in_channels == module_params['in_channels'], print(
                            prev_in_channels, module_params
                        )
                else:
                    module_params['in_channels'] = prev_in_channels
                if module_params.get('out_channels', False):
                    pass
                else:
                    module_params['out_channels'] = out_channels
                    out_channels = None
                prev_in_channels = module_params['out_channels']
            elif c == 'Linear':
                if module_params.get('in_features', False):
                    if prev_in_channels is not None:
                        assert prev_in_channels == module_params['in_features']
                else:
                    module_params['in_features'] = prev_in_channels
                if module_params.get('out_features', False):
                    pass
                else:
                    module_params['out_features'] = out_channels
                prev_in_channels = module_params['out_features']
            ModuleCls = getattr(nn, c)
            m = ModuleCls(**module_params)
            modules.append(m)
    elif hasattr(network_modules_params, '__call__'):
        return network_modules_params(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError
    return modules


def build_scheduler(scheduler_params):
    import dsb.schedulers

    c, scheduler_params = parse_params(scheduler_params)
    SchedulerCls = getattr(dsb.schedulers, c)
    return SchedulerCls(**scheduler_params)


def build_storage(storage_params, *storage_args, **kwargs):
    import dsb.buffers.storage

    c, storage_params = parse_params(storage_params)
    StorageCls = getattr(dsb.buffers.storage, c)
    return StorageCls(*storage_args, **storage_params, **kwargs)


def build_buffer(buffer_params, obs_space, action_space, tmp_dir=None, **buffer_kwargs):
    import dsb.buffers
    import dsb.buffers.buffer_wrapper

    c, buffer_params = parse_params(buffer_params)

    storage_params = buffer_params.pop('storage_params', None)
    if storage_params is not None:
        storage = build_storage(
            storage_params,
            obs_space,
            action_space,
            tmp_dir=tmp_dir,
        )
        storage.allocate()
    else:
        from dsb.buffers.storage import BaseStorage

        storage = BaseStorage(obs_space, action_space, None, tmp_dir=tmp_dir)

    buffer_wrappers = buffer_params.pop('buffer_wrappers', [])
    BufferCls = getattr(dsb.buffers, c)
    buffer = BufferCls(storage, **buffer_params, **buffer_kwargs)
    for buffer_wrapper_param in buffer_wrappers:
        assert 'cls' in buffer_wrapper_param.keys()
        w_c, wrapper_params = parse_params(buffer_wrapper_param)
        WrapperCls = getattr(dsb.buffers.buffer_wrapper, w_c)
        buffer = WrapperCls(buffer, **wrapper_params, **buffer_kwargs)
    return buffer


def build_normalizer(normalizer_params, obs_space):
    import dsb.normalizers
    from dsb.normalizers import DictNormalizer, IdentityNormalizer

    c, normalizer_params = parse_params(normalizer_params)

    if c == 'DictNormalizer':
        _normalizers = {}
        _normalizers_params = {}

        for k, v in obs_space.spaces.items():
            p = normalizer_params.get(k, -1)
            if isinstance(p, dict):
                p = p.copy()
                if 'cls' in p.keys():
                    _normalizers[k] = build_normalizer(p, v)
                    _normalizers_params[k] = {}
                elif 'which' in p.keys():  # allow different keys to share the same normalizer
                    which = p.pop('which')
                    _normalizers[k] = _normalizers[which]
                    _np = _normalizers_params[which].copy()
                    _np.update(p)
                    _normalizers_params[k] = _np
                else:
                    raise ValueError
            elif p == -1:
                _normalizers[k] = IdentityNormalizer(v)
                _normalizers_params[k] = dict(update_with=False)
            else:
                raise ValueError

        for k in obs_space.spaces.keys():
            d = normalizer_params.get(k, {})
            if isinstance(d.get('which', False), str):
                assert _normalizers[k] is _normalizers[d['which']]  # check if same object

        if normalizer_params.get('embedding_target', False):
            _normalizers['embedding_target'] = build_normalizer(
                normalizer_params['embedding_target'], obs_space['achieved_goal']
            )
            _normalizers_params['embedding_target'] = dict(update_with=False)

        return DictNormalizer(_normalizers, normalizers_params=_normalizers_params)
    else:
        NormalizerCls = getattr(dsb.normalizers, c)

    return NormalizerCls(obs_space, **normalizer_params)


def build_policy(policy_list_params, agent, **kwargs):
    import dsb.policies

    if not isinstance(policy_list_params, list):
        policy_list_params = [policy_list_params]

    policy = agent
    for policy_params in policy_list_params:
        c, policy_params = parse_params(policy_params)
        PolicyCls = getattr(dsb.policies, c)
        policy = PolicyCls(policy, **policy_params, **kwargs)
    return policy


def build_agent(agent_params, *args, **kwargs):
    import dsb.agents

    c, agent_params = parse_params(agent_params)
    AgentCls = getattr(dsb.agents, c)
    return AgentCls(*args, **agent_params, **kwargs)


def build_embedding_head(
    embedding_head_params,
    *args,
    embedding_head_wrappers=None,
    **kwargs,
):
    import dsb.embedding_heads

    c, embedding_head_params = parse_params(embedding_head_params)
    EmbeddingHeadCls = getattr(dsb.embedding_heads, c)
    embedding_head = EmbeddingHeadCls(*args, **embedding_head_params, **kwargs)

    if embedding_head_wrappers is not None:
        for embedding_head_wrapper_params in embedding_head_wrappers:
            c, embedding_head_wrapper_params = parse_params(embedding_head_wrapper_params)
            EmbeddingHeadWrapperCls = getattr(dsb.embedding_heads, c)
            embedding_head = EmbeddingHeadWrapperCls(
                embedding_head, **embedding_head_wrapper_params
            )
    return embedding_head


def build_dynamics_head(dynamics_head_params, mdp_space, **kwargs):
    import dsb.dynamics_heads

    c, dynamics_head_params = parse_params(dynamics_head_params)
    DynamicsHeadCls = getattr(dsb.dynamics_heads, c)
    dynamics_head = DynamicsHeadCls(mdp_space, **dynamics_head_params, **kwargs)
    return dynamics_head


def clone_module(module, reset_parameters=True):
    # TODO: use the appropriate build function so allow different parameters to be specified?
    cloned = copy.deepcopy(module)
    if reset_parameters:
        cloned.reset_parameters()
    return cloned


def build_eval_func(eval_func_params):
    import dsb.runner

    def eval_func(*args, **kwargs):
        o = {}
        for eval_id, p in eval_func_params.items():
            p = p.copy()
            c = p.pop('cls')
            eval_func = getattr(dsb.runner, c)
            eval_info = eval_func(eval_id, *args, **kwargs, **p)
            o.update(eval_info)
        return o

    if eval_func_params is not None:
        return eval_func
    else:
        return None


def get_env_load_fn(env_type):
    import dsb.envs

    if env_type == 'gym':
        return gym.make
    else:
        return getattr(dsb.envs, env_type).env_load_fn


def module_reset_parameters(m):
    reset_parameters = getattr(m, 'reset_parameters', None)
    if callable(reset_parameters):
        m.reset_parameters()
