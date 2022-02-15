from dsb.dependencies import *

import os
import psutil
import gc


# https://discuss.pytorch.org/t/how-pytorch-releases-variable-garbage/7277/2
def gc_obj_report(tensors_only=False):
    c_obj = collections.defaultdict(int)
    c_tensor = collections.defaultdict(int)
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tag = str(type(obj)) + str(obj.size())
                c_tensor[tag] += 1
            else:
                tag = type(obj)
                c_obj[tag] += 1

        except ReferenceError as e:  # ReferenceError: weakly-referenced object no longer exists
            pass

    if not tensors_only:
        print(c_obj)
    print('-' * 10)
    print(c_tensor)
    print(f'total num tensors = {sum([v for v in c_tensor.values()])}')


def memory_usage_report(pid):
    mem = psutil.virtual_memory()
    proc = psutil.Process(pid)
    proc_used = proc.memory_info()[0]
    mem_usage = dict(
        total=mem.total,
        available=mem.available,
        shared=mem.shared,
        proc_used=proc_used,
        perc_proc_used=proc_used / mem.total,
    )
    return mem_usage


import subprocess
import xml.etree.ElementTree


def get_process_gpu_usage(pid):
    try:
        cmd_out = subprocess.check_output(['nvidia-smi', '-q', '-x'], timeout=5)  # in seconds
        gpu_processes = xml.etree.ElementTree.fromstring(cmd_out).findall('.//process_info')
        for p in gpu_processes:
            if int(p.find('pid')) == pid:
                return int(p.find('used_memory').text.split(' ')[0])
    except Exception:
        pass
    return None
