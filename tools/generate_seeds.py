import random
import sys

random.seed()

# seed range is (0, 2**32 - 1)
# lets leave some space for seed + n, parallel envs
a = 0
b = 2 ** 31 - 1

n = int(sys.argv[-1])
seeds = [random.randint(a, b) for _ in range(n)]
print(seeds)
