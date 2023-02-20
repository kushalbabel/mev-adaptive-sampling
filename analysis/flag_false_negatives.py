import json
import argparse
from collections import defaultdict
import glob
import sys

problemsdir = '/home/kb742/mev-adaptive-sampling/eth_token_tests_uniswapv3/*/*/amm_reduced'

parser = argparse.ArgumentParser(description='Flag false negatives')
parser.add_argument('-d', '--data', help="Input File path containing raw data")

demanded_data = json.load(open('../data/flashbots_data_on_demand.json', 'r'))

def get_baseline_data(block_number):
    if block_number < 11800000:
        return None
    if block_number <= 14986955:
        for block in data:
            if block['block_number'] == block_number:
                return block
        return None
    else:
        if str(block_number) in demanded_data:
            block = demanded_data[str(block_number)]
            if block == {}:
                return None
            else:
                return block 
        else:
            return None
        

def is_false_negative(problem_file):
    fp = open(problem_file, 'r')
    lines = fp.readlines()
    if len(lines) == 0:
        return False
    block_number = int(lines[0].strip().split(',')[0])
    
    block = get_baseline_data(block_number)
    if block is None:
        return False

    problem_transactions = set()
    for line in lines[1:]:
        if line.startswith('0,'):
            problem_transactions.add(line.strip().split(',')[-1])

    bundles = defaultdict(lambda: [])
    involved_bundles = set()

    
    for tx in block["transactions"]:
        bundles[tx['bundle_index']].append(tx)
    
    for idx in bundles:
        for tx in bundles[idx]:
            if tx['transaction_hash'] in problem_transactions:
                involved_bundles.add(idx)
    
    for idx in involved_bundles:
        for tx in bundles[idx]:
            if tx['transaction_hash'] not in problem_transactions:
                return True
        
    return False

args = parser.parse_args()
f = open(args.data, 'r')
data = json.load(f)

for filename in glob.glob(problemsdir):
    false_negative = is_false_negative(filename)
    if false_negative:
        print('{}'.format(filename))
        sys.stdout.flush()
