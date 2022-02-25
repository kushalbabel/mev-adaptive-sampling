import os
from pathlib import Path
from collections import defaultdict
import json
import requests
import argparse
import logging
from subprocess import Popen, PIPE
import pandas as pd
import re
import random

def to_format(transaction):
    return '0,{},{}\n'.format(transaction['from'], transaction['hash'])


parser = argparse.ArgumentParser(description='Generate problems')

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
    '-sb', '--start-block',
    help="StartBlock",
    required=True
)

parser.add_argument(
    '-eb', '--end-block',
    help="EndBlock",
    required=True
)

parser.add_argument(
    '-o', '--output-dir',
    help="output dir",
    required=True
)


ARCHIVE_NODE_URL = 'http://localhost:8545'
def query_block(block_number):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getBlockByNumber'
    data['params'] = [block_number, True] # get full tx
    data['id'] = block_number + 1000000
    r = requests.post(ARCHIVE_NODE_URL, json=data)
    response = json.loads(r.content)
    return response

args = parser.parse_args()

output = ''

for block in range(int(args.start_block), int(args.end_block)):
    pipe = Popen("grep 'block " + str(block) + "' " + args.file, shell=True, stdout=PIPE, stderr=PIPE)
    output += str(pipe.stdout.read() + pipe.stderr.read(), "utf-8")

lines = output.strip().split('\n')
block_to_tx = defaultdict(lambda : set())
for line in lines:
    if line.startswith('//') and len(line) > 0:
        transaction_hash = line.split()[2]
        block_num = int(line.split()[4])
        block_to_tx[block_num].add(transaction_hash)


# swap_template1 = '3,Miner,usdc,alpha1,eth,0,usdc,0,0'
# swap_template2 = '3,Miner,eth,alpha2,usdc,0,eth,0,0'
# addition_template = '1,Miner,alpha3,eth,alpha4,usdc'
# removal_template = '4,Miner,alpha5,eth,usdc'

# insertions = [swap_template1, swap_template2, addition_template, removal_template]

dir = args.output_dir

for block in block_to_tx:
    print(block)
    filename1 = '{}/{}/amm'.format(dir, block)
    filename2 = '{}/{}/amm_reduced'.format(dir, block)
    os.makedirs(os.path.dirname(filename1), exist_ok=True)
    os.makedirs(os.path.dirname(filename2), exist_ok=True)
    f1 = open(filename1.format(), 'w')
    f2 = open(filename2.format(), 'w')
    necessary_transactions = block_to_tx[block]
    interacting_addresses = set()
    complete_block = query_block(block)
    complete_output = ''
    reduced_output = ''
    all_transactions = complete_block['result']['transactions']
    for tx in all_transactions:
        if tx['hash'] in necessary_transactions:
            interacting_addresses.add(tx['from'])
            interacting_addresses.add(tx['to'])
    f1.write('{}\n'.format(block))
    f2.write('{}\n'.format(block))
    for tx in all_transactions:
        f1.write(to_format(tx))
        if tx['from'] in interacting_addresses or tx['to'] in interacting_addresses:
            f2.write(to_format(tx))