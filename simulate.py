import argparse
import logging
from uniswapv2 import UniswapV2

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

def get_mev(amm):
    mev = 0
    miner_balances = amm.config()['Miner']
    amm_reserves = amm.config()[amm.exchange_name]
    default_token = 'eth'
    # first convert all into eth
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
                return large_negative

    for token in miner_balances:
        amount = miner_balances[token]
        if token == default_token:
            mev += amount
        elif token == amm.lp_token:
            mev += 2*amm.config()[amm.exchange_name][default_token]*amount/amm.supply
        else:
            pass
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
