import argparse
import logging
from uniswapv2 import UniswapV2
from copy import deepcopy

large_negative = -1e9

def info(amm):
    print(dict(amm.config()))
    print('Supply',amm.supply)

def generate_transaction(tx_type, params):
    if tx_type == '1':
        format_string = '{} adds {} {} and {} {} of liquidity'
    elif tx_type == '2':
        format_string = '{} removes {} {} and {} {} of liquidity'
    elif tx_type == '3':
        format_string = '{} swaps for {} by providing {} {} and {} {} with change {} fee {}'
    elif tx_type == '4':
        format_string = '{} redeems {} fraction of liquidity from {} and {}'
    return format_string.format(*params)

def get_mev(amm_orig):
    mev = 0
    amm = deepcopy(amm_orig)
    miner_balances = amm.config()['Miner']
    default_token = 'eth'
    # first convert all into eth
    amount = miner_balances[amm.lp_token]
    amm.raw_redeem('Miner', amount, 'eth', 'usdc')
        
    miner_balances = amm.config()['Miner']
    for token in miner_balances:
        amount = miner_balances[token]
        if token == default_token:
            pass
        elif token == amm.lp_token:
            pass
        else:
            if amount > 0:
                # swap into eth_amount, not just mul price
                amm.raw_swap('Miner', token, default_token, amount, 0, 0)
            else:
                amm.raw_swap_output('Miner', default_token, token, 0-amount)
                #return large_negative
    
    mev = amm.config()['Miner'][default_token]
    return mev

def simulate(lines):
    amm = UniswapV2(balances={'eth':0,'usdc':0})
    for line in lines:
        elements = line.strip().split(',')
        function_selector = elements[0]
        transaction = generate_transaction(function_selector, elements[1:])
        amm.process(transaction)
        # print(line)
        # info(amm)
        # print('mev',get_mev(amm))
        # print('')
    return get_mev(amm)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run UniswapV1 experiments')

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

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel, format='%(message)s')

    logger = logging.getLogger(__name__)

    data_f = open(args.file, 'r')
    mev = simulate(data_f.readlines())
    print(mev)
