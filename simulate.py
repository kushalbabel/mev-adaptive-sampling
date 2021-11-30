import argparse
import logging
from uniswapv2 import UniswapV2
        
def generate_transaction(tx_type, params):
    if tx_type == '1':
        format_string = '{} adds {} {} and {} {} of liquidity'
    elif tx_type == '2':
        format_string = '{} removes {} {} and {} {} of liquidity'
    elif tx_type == '3':
        format_string = '{} swaps for {} by providing {} {} and {} {} with change {} fee {}'
    return format_string.format(*params)

def simulate(lines):
    amm = UniswapV2(balances={'eth':0,'usdc':0})
    for line in lines:
        elements = line.strip().split(',')
        function_selector = elements[0]
        transaction = generate_transaction(function_selector, elements[1:])
        amm.process(transaction)

    mev = 0
    miner_balances = amm.config()['Miner']
    amm_reserves = amm.config()['UniswapV2']
    default_token = 'eth'
    for token in miner_balances:
        amount = miner_balances[token]
        if token == default_token:
            mev += amount
        else:
            eth_amount = amount * amm_reserves[default_token]/amm_reserves[token]
            mev += eth_amount
    return mev

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