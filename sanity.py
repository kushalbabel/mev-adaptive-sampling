#!/usr/bin/env python3
from simulate_foundry import simulate
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support

data_f = open('manualtests/temp3', 'r')
lines = data_f.readlines()
with Pool() as pool:
    result = pool.starmap(simulate, [(lines,x) for x in range(24)])
    print(result)