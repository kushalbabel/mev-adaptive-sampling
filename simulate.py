import argparse
import logging
from uniswapv2 import UniswapV2

parser = argparse.ArgumentParser(description='Run UniswapV1 experiments')

parser.add_argument(
    '-v', '--verbose',
    help="Be verbose",
    action="store_const", dest="loglevel", const=logging.INFO,
    default=logging.WARNING
)

parser.add_argument(
    '-e', '--exchange',
    help="uniswapv1",
    default='uniswapv1'
)


parser.add_argument(
    '-f', '--file',
    help="Input File path",
    required=True
)

parser.add_argument(
    '-o', '--output',
    help="Output File path",
    required=True
)

def generate_transaction(tx_type, params):
    if tx_type == '1':
        format_string = '{} adds {} {} and {} {} of liquidity'
    elif tx_type == '2':
        format_string = '{} removes {} {} and {} {} of liquidity'
    elif tx_type == '3':
        format_string = '{} swaps for {} by providing {} {} and {} {} with change {} fee {}'
    return format_string.format(*params)
        

args = parser.parse_args()    
logging.basicConfig(level=args.loglevel, format='%(message)s')

logger = logging.getLogger(__name__)

data_f = open(args.file, 'r')

output_f = open(args.output, 'w')

amm = UniswapV2(balances={'eth':0,'link':0})

for line in data_f.readlines():
    elements = line.split(',')
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

print(mev)

