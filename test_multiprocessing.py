import multiprocessing as mp
import matplotlib.pyplot as plt

from itertools import repeat

import sys
sys.path.append('backend/')

# from simulate import simulate
from simulate_efficient_hardhat import simulate, setup, prepare_once
data_f = open('/home/kb742/mev-adaptive-sampling/manualtests/optimised_2', 'r')
lines = data_f.readlines()
PARALLEL = 4
PORTS = [i for i in range(PARALLEL)]
results = [-1 for i in range(PARALLEL)]
dexes = ['uniswapv2','uniswapv3']

with mp.Pool() as pool:
    ctx = setup(lines[0])
    print(ctx.decimals)
    ctxes = pool.starmap(prepare_once, zip(repeat(ctx), repeat(lines), PORTS, repeat(dexes)))
    print(ctx.decimals)
    results = pool.starmap(simulate, zip(ctxes, repeat(lines), PORTS, repeat(dexes)))

    print(results)
    