# domain-shift-benchmark

## Quickstart

You'll probably want to first install [`KitchenShift`](https://github.com/etaoxing/kitchen-shift), the environment we use in our paper, forked from [`adept_envs`](https://github.com/google-research/relay-policy-learning/blob/master/adept_envs/adept_envs/franka/kitchen_multitask_v0.py).

```
pip install -r requirements.txt
pip install -e .
```

To run behavioral cloning:

```
python tools/run_experiment.py exp_compare_bc -p experiments/domain_shift_benchmark/ -s 0 1 2 3 -w 0 1 2 3 -v 0
```

This starts 4 runs on each of 4 GPUs for variant 0 from [`exp_compare_bc.py`](experiments/domain_shift_benchmark/exp_compare_bc.py), see the `_VARIANTS_` variable in the experiment file for a list of the models. The random seeds used are specified by `_SEEDS_`. The run logs will be located in `_EXP_DIR + /{MODEL}/run_seed{SEED}`. 

## References
[1] Our paper
```
@inproceedings{xing2021kitchenshift,
    title={KitchenShift: Evaluating Zero-Shot Generalization of Imitation-Based Policy Learning Under Domain Shifts},
    author={Xing, Eliot and Gupta, Abhinav and Powers*, Sam and Dean*, Victoria},
    booktitle={NeurIPS 2021 Workshop on Distribution Shifts: Connecting Methods and Applications},
    year={2021},
    url={https://openreview.net/forum?id=DdglKo8hBq0}
}
```