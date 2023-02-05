import json
import argparse
from collections import defaultdict
import glob
import sys

problemsdir = '/home/kb742/mev-adaptive-sampling/eth_token_tests_uniswapv2/*/*/amm_reduced'

parser = argparse.ArgumentParser(description='Flag false negatives')
parser.add_argument('-d', '--data', help="Input File path containing raw data")
# parser.add_argument('-o', '--outfile', help="output csv")
# parser.add_argument('-p', '--problemsdir', help="Problem Dir")

def is_false_negative(problem_file, data_json):
    fp = open(problem_file, 'r')
    lines = fp.readlines()
    if len(lines) == 0:
        return False
    block_number = int(lines[0].strip().split(',')[0])
    if block_number < 11800000:
        return False
    problem_transactions = set()
    for line in lines[1:]:
        if line.startswith('0,'):
            problem_transactions.add(line.strip().split(',')[-1])

    bundles = defaultdict(lambda: [])
    involved_bundles = set()

    for block in data:
        # print(block['block_number'])
        if block['block_number'] > block_number:
            continue
        elif block['block_number'] < block_number:
            break
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
print('data reading completed.')

for filename in glob.glob(problemsdir):
    # print(filename)
    false_negative = is_false_negative(filename, data)
    if false_negative:
        print('{}'.format(filename))
        sys.stdout.flush()
