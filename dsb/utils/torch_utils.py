from dsb.dependencies import *


def load_pretrained(
    module,
    state_dict_path,
    state_dict_key=None,
    device='cpu',
    freeze=False,
):
    # optimizer load_state_dict device issue, https://github.com/pytorch/pytorch/issues/2830
    # could also do something like https://github.com/allenai/allennlp/pull/1144/files
    module.to(device)
    state_dict = torch.load(state_dict_path, map_location=device)

    if state_dict_key is not None:
        s = state_dict[state_dict_key]
    else:
        s = state_dict

    module.load_state_dict(s)
    if freeze:
        module.optimize_interval = None

        for param in module.parameters():
            param.requires_grad = False

    return state_dict


def stop_grad(x):  # convenience function for detaching tensor or dict of tensors
    if isinstance(x, torch.Tensor):
        return x.detach()
    elif isinstance(x, dict):
        y = {}
        for k, v in x.items():
            if k[0] == '_':
                y[k] = v
            else:
                y[k] = v.detach()
        return y
    else:
        raise ValueError(f'unsupported type {type(x)}')


DEVICE = 'cpu'
XLA = False


def torchify(
    x,
    dtype=torch.float32,
    non_blocking=True,
    device=None,
    pin_memory=False,
    copy=False,
):
    if device is None:
        device = DEVICE

    if isinstance(x, torch.Tensor):
        y = x.to(dtype=dtype, device=device, non_blocking=non_blocking)
        if pin_memory and device == 'cpu' and torch.cuda.is_available():
            y = y.pin_memory()

    # elif isinstance(x, np.ndarray) and dtype == None:
    #     y = torch.from_numpy(x)
    #     if pin_memory and device == 'cpu' and torch.cuda.is_available():
    #         y = y.pin_memory()

    elif isinstance(x, np.ndarray) or isinstance(x, list):
        if copy:
            y = torch.tensor(x, dtype=dtype, device=device)
        else:
            y = torch.as_tensor(x, dtype=dtype, device=device)
        if pin_memory and device == 'cpu' and torch.cuda.is_available():
            y = y.pin_memory()

    elif isinstance(x, dict):
        y = {}
        for k, v in x.items():
            if k[0] == '_':
                y[k] = v
            else:
                y[k] = torchify(
                    v,
                    dtype=dtype,
                    non_blocking=non_blocking,
                    device=device,
                    pin_memory=pin_memory,
                    copy=copy,
                )

    elif x is None:
        return x

    else:
        raise ValueError(f'unsupported type {type(x)}')

    return y


def untorchify(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    elif isinstance(x, dict):
        return {k: untorchify(v) for k, v in x.items()}
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, float):
        return x
    else:
        raise ValueError(f'unsupported type {type(x)}')


def set_device(d):
    global DEVICE
    DEVICE = d
    if d.type == 'cuda':
        torch.cuda.set_device(d)  # see https://github.com/pytorch/pytorch/pull/4196


def set_xla(on):
    global XLA
    XLA = on

    import torch_xla.core.xla_model as xm

    torch.save = xm.save


def module_repr_include(repr, to_include_dict):
    s_arr = repr.split('\n')
    for i, k in enumerate(to_include_dict.keys()):
        s_arr.insert(1 + i, f'  ({k}): {to_include_dict[k]}')
    s = '\n'.join(s_arr)
    return s


# from https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/utils.py#L10
# TODO: nested check if any module in agent is eval
class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)
# if not getattr(nn, 'Flatten', False): # for torch <=1.2.0
#     nn.Flatten = Flatten


# Compute shape by doing one forward pass
def compute_output_shape(x, module, aug=None):
    with torch.no_grad():
        # assume x is unbatched
        x = torchify(x).unsqueeze(0)

        if aug:
            x = aug(x)

        h = module(x).squeeze(0)
        conv_output_shape = tuple(h.shape)
        n_flatten = h.flatten().shape[0]
        return conv_output_shape, n_flatten
