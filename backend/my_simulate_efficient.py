from unicodedata import decimal
import requests
import json
import argparse
import sys
import logging
from copy import deepcopy
from contracts import utils
from contracts.uniswap import uniswap_router_contract, sushiswap_router_contract, uniswapv3_router_contract, uniswapv3_quoter_abi, position_manager_abi, position_manager_path
from contracts.tokens import erc20_abi, weth_abi
from web3 import Web3
from collections import defaultdict
import logging
import time
from datetime import datetime
from utils import get_price
import rlp
import eth_abi
from eth_utils import keccak, to_checksum_address, to_bytes

import simulate_efficient_hardhat

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Simulation on a mainnet fork')

    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
        default=logging.WARNING
    )

    parser.add_argument(
        '-f', '--file',
        help="Input File path",
        required=True
    )

    parser.add_argument(
        '-p', '--port',
        help="Id of one of the many backend client",
        required=False,
        default=-1
    )

    parser.add_argument(
        '-s', '--settlement',
        help="cex/dex/max",
        required=False,
        default='max'
    )

    args = parser.parse_args()
    
    data_f = open(args.file, 'r')
    port_id = int(args.port)
    lines = data_f.readlines()
    print("setting up...", lines[0])
    ctx_initial = simulate_efficient_hardhat.setup(lines[0])
    
    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()
    for i in range(5):
        print("Round #", i);
        total_time = 0
        current = time.time()
        ctx = simulate_efficient_hardhat.prepare_once(ctx_initial, lines, port_id, ['uniswapv2', 'uniswapv3'])
        print("simulating...")
        mev = simulate_efficient_hardhat.simulate(ctx, lines, port_id, ['uniswapv2', 'uniswapv3'], False, '', args.settlement)
        print(mev)
        print("single run time: ", time.time() - current)
    
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
