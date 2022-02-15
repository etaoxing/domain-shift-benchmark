import sys
import argparse
import os
import itertools
import importlib
import copy
from functools import partial
from tqdm import tqdm

from dsb.utils import update_cfg


parser = argparse.ArgumentParser()
parser.add_argument('cfg', type=str)
parser.add_argument('-p', '--expfile_dir', default='experiments/', type=str)
# parser.add_argument('-e', '--exp', type=str, default='test', choices=['train', 'test'])
# parser.add_argument('-r', '--run', type=str, default='')
parser.add_argument('-s', '--seed_idx', type=int, default=None, nargs='*')
parser.add_argument(
    '-w', '--workers', type=int, default=[-1], nargs='*'
)  # specify gpu to use for a worker process
parser.add_argument('-v', '--variants', default=None, type=int, nargs='*')
parser.add_argument('-vs', '--variants_str', default=None, type=str, nargs='*')
parser.add_argument('--dry_run', action='store_true')
args = parser.parse_args()
if args.variants is not None:
    args.variants = set(args.variants)
if args.variants_str is not None:
    args.variants_str = set(args.variants_str)

sys.path.insert(0, args.expfile_dir)
exp_cfg = importlib.import_module(args.cfg)

os.makedirs(exp_cfg._EXP_DIR, exist_ok=True)

# generate all experiment config variants
from dsb.utils import load_cfg, AttrDict

base_cfg = load_cfg(os.path.join(args.expfile_dir, exp_cfg._BASE_))

cfg_files = []
for i, variant in enumerate(exp_cfg._VARIANTS_):
    v = update_cfg(base_cfg, variant)
    v = AttrDict(**v)

    t = v.ckpt_dir.split('/')[-2]
    f = os.path.join(exp_cfg._EXP_DIR, f'config_{t}.py')
    v.save_to(f)

    print(i, f)

    if args.variants is not None:
        if i not in args.variants:
            continue
    elif args.variants_str is not None:
        ckpt_dir_split = v['ckpt_dir'].split('/')
        if ckpt_dir_split[-1] == '':
            check_id = ckpt_dir_split[-2]
        else:
            check_id = ckpt_dir_split[-1]

        if check_id not in args.variants_str:
            continue

    cfg_files.append(f)

# take subset of seeds
if args.seed_idx is not None:
    exp_cfg._SEEDS_ = [exp_cfg._SEEDS_[i] for i in args.seed_idx]

experiments = list(sorted(itertools.product(exp_cfg._SEEDS_, cfg_files), key=lambda x: x[1]))
print('experiments to run: ', experiments)

if args.dry_run:
    exit()


def run_cfg(seed, cfg, gpu):
    run = f'run_seed{seed}'
    cmd = f'MUJOCO_GL=egl EGL_DEVICE_ID={gpu} python -u -O tools/run.py {cfg} -e train --run {run} --seed {seed} --gpu {gpu}'
    out = os.system(cmd + " 1>/dev/null")
    return out


# run experiments
if len(args.workers) == 1:
    print('running w/o threadpool')

    pbar = tqdm(experiments)
    for seed, cfg in pbar:
        pbar.set_description(f"{cfg.split('/')[-1]}, {seed}")
        run_cfg(seed, cfg, args.workers[0])

else:
    print('running with threadpool')

    def _run_cfg(x, q=None):
        seed, cfg = x
        w = q.get()

        job = f"cfg={cfg.split('/')[-1]}, seed={seed}, worker={w}"

        print(f'starting, job: {job}')
        out = run_cfg(seed, cfg, w)
        print(f'done {out}, job: {job}')
        q.put(w)
        return out

    from queue import Queue
    from concurrent.futures import ThreadPoolExecutor, as_completed

    q = Queue(maxsize=len(args.workers))
    for w in args.workers:
        q.put(w)
    with ThreadPoolExecutor(max_workers=len(args.workers)) as executor:
        futures = [executor.submit(partial(_run_cfg, q=q), e) for e in experiments]
        for f in tqdm(as_completed(futures), total=len(futures)):
            f.result()
