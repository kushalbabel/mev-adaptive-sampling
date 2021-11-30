import argparse
import logging
from subprocess import Popen, PIPE
import pandas as pd
import re
import random

def to_csv(transaction):
    tokens = {'1097077688018008265106216665536940668749033598146' : 'eth' , '917551056842671309452305380979543736893630245704':'usdc'}
    decimals = {'1097077688018008265106216665536940668749033598146' : 18 , '917551056842671309452305380979543736893630245704':6}
    if 'adds' in transaction:
        tx_type = 1
    elif 'removes' in transaction:
        tx_type = 2
    else:
        tx_type = 3
    if tx_type == 1 or tx_type == 2:
        if tx_type == 1:
            vals = re.match(r'(.*) adds (.*) (.*) and (.*) (.*) of liquidity;', transaction)
        else:
            vals = re.match(r'(.*) removes (.*) (.*) and (.*) (.*) of liquidity;', transaction)
        token1 = vals.group(3)
        token2 = vals.group(5)
        return '{},{},{:.4f},{},{:.4f},{}'.format(tx_type, vals.group(1), 1.0*float(vals.group(2))/(10**decimals[token1]), tokens[token1], 1.0*float(vals.group(4))/(10**decimals[token2]), tokens[token2])
    else:
        vals = re.match(r'(.*) swaps for (.*) by providing (.*) (.*) and (.*) (.*) with change (.*) fee (.*) ;', transaction)
        token1 = vals.group(4)
        token2 = vals.group(6)
        return '{},{},{},{:.4f},{},{:.4f},{},{:.4f},{}'.format(tx_type, vals.group(1), tokens[vals.group(2)], 1.0*float(vals.group(3))/(10**decimals[token1]), tokens[token1], 
            1.0*float(vals.group(5))/(10**decimals[token2]), tokens[token2], 1.0*float(vals.group(7))/(10**decimals[token1]), vals.group(8))



parser = argparse.ArgumentParser(description='Generate Sushiswap problems')

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
    '-r', '--reserves',
    help="reserves file",
    required=True
)

parser.add_argument(
    '-sb', '--start-block',
    help="Block",
    required=True
)

parser.add_argument(
    '-eb', '--end-block',
    help="Block",
    required=True
)



args = parser.parse_args()

output = ''

for block in range(int(args.start_block), int(args.end_block)):
    pipe = Popen("grep -A1 'block " + str(block) + "' " + args.file, shell=True, stdout=PIPE, stderr=PIPE)
    output += str(pipe.stdout.read() + pipe.stderr.read(), "utf-8")

lines = output.strip().split('\n')
transactions = []
for line in lines:
    if not line.startswith('//') and len(line) > 0:
        transactions.append(line)

csv_transactions = []
for transaction in transactions:
    csv_transactions.append(to_csv(transaction))    

df = pd.read_csv(args.reserves)
df = df[(df.Block) < int(args.start_block)].iloc[-1]
bootstrap_tx = '1,Reserves,{},usdc,{},eth'.format(df.Reserve0,df.Reserve1)

swap_template1 = '3,Miner,usdc,alpha1,eth,0,usdc,0,0'
swap_template2 = '3,Miner,eth,alpha2,usdc,0,eth,0,0'
addition_template = '1,Miner,alpha3,eth,alpha4,usdc'
removal_template = '2,Miner,alpha5,eth,alpha6,usdc'

insertions = [swap_template1, swap_template2, addition_template, removal_template]

mempool_transactions = csv_transactions + insertions
random.shuffle(mempool_transactions)
final_transactions =  [bootstrap_tx] + mempool_transactions

for tx in final_transactions:
    print(tx)
