from dsb.dependencies import *

import pprint
import ast


def from_nested_dict(data):
    if not isinstance(data, dict):
        return data
    else:
        return AttrDict({key: from_nested_dict(data[key]) for key in data})


class AttrDict(dict):
    def __init__(self, *d, **kwargs):
        super().__init__(*d, **kwargs)
        self.__dict__ = self
        for key in self.keys():
            self[key] = from_nested_dict(self[key])

    def __str__(self):
        try:
            return pprint.pformat(self, sort_dicts=False)
        except Exception as e:  # sort_dicts only available for python3.8+
            return pprint.pformat(self)

    def save_to(self, filepath):
        f = open(filepath, 'w')
        f.write(self.__str__())
        # f.write(super().__str__())


def load_cfg(file):
    # TODO: make this safer with ast.literal_eval
    return AttrDict(**eval(open(file, 'r').read()))


def copy_cfg(cfg):
    # need to recreate AttrDict, otherwise can't access attributes
    return AttrDict(copy.deepcopy(cfg))


def update_cfg(d, u, inplace=False):
    if not inplace:
        d = copy.deepcopy(d)
        u = copy.deepcopy(u)

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            _v = update_cfg(d.get(k, {}), v, inplace=inplace)
        else:
            _v = v
        d[k] = _v
    return d


def cfg_override_extras(cfg, extras):
    if len(extras) > 0:
        if extras[0] != '--cfg_override':
            raise ValueError('Unrecognized arg extras.')

        # print(extras)
        i = 1
        while i < len(extras):
            k = extras[i]
            v = extras[i + 1]

            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                # assume string
                pass

            p = k.split('.')
            a = cfg
            for _p in p[:-1]:
                a = a[_p]
            a[p[-1]] = v
            # print(k, v, type(v))

            i += 2
    return cfg
