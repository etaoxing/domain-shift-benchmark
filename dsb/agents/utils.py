from dsb.dependencies import *
from dsb.utils import torchify


# polyak averaging
def update_target_network(network, target_network, tau=1.0):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling
# this comes from SoRB, implemented in tf
def variance_initializer_(tensor, scale=1.0, mode='fan_in', distribution='uniform'):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        scale /= max(1.0, fan_in)
    elif mode == "fan_out":
        scale /= max(1.0, fan_out)
    else:
        raise ValueError

    if distribution == 'uniform':
        limit = math.sqrt(3.0 * scale)
        nn.init.uniform_(tensor, -limit, limit)
    else:
        raise ValueError


def min_max_normalize(x):  # scale b/w 0 and 1
    shape = x.shape
    x = x.view(x.size(0), -1)
    x -= x.min(1, keepdim=True)[0]
    x /= x.max(1, keepdim=True)[0]
    x = x.view(shape)
    return x


def relabel_goal_for_transition(buffer, _next_state, reward, done, next_state_embedding=None):
    pass


def get_pair(buffer, ptrs, try_cached=False, save_cached=True):
    # if try_cached:
    #     had_cached, result = buffer.get_cached('pair')
    #     if had_cached:
    #         return result
    # # if cannot grab cached, then embedding_head did not sample_pairs

    ptrs_pair, state_pair, distance_pair, pos_mask = buffer.sample_pair_given(ptrs)
    state_pair = torchify(state_pair)

    # setting dtype=None to keep same dtype
    distance_pair, pos_mask = torchify(distance_pair, dtype=None), torchify(pos_mask, dtype=None)
    # NOTE: pair order matters, may not be the same size as ptrs

    result = (state_pair, distance_pair, pos_mask)

    # if save_cached:
    #     buffer.save_in_cache('pair', result)

    return result
