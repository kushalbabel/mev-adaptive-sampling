import json
import argparse
from collections import defaultdict
import glob
import sys

problemsdir = '/home/kb742/mev-adaptive-sampling/eth_token_tests_uniswapv3/*/*/amm_reduced'

parser = argparse.ArgumentParser(description='Run Optimization')
parser.add_argument('-d', '--data', help="Input File path containing raw data")
# parser.add_argument('-p', '--problemsdir', help="Problem Dir")

def get_baseline_problem(problem_file, data_json):
    fp = open(problem_file, 'r')
    lines = fp.readlines()
    if len(lines) == 0:
        return -1
    block_number = int(lines[0].strip().split(',')[0])
    if block_number < 11800000:
        return -1
    problem_transactions = set()
    for line in lines[1:]:
        if line.startswith('0,'):
            problem_transactions.add(line.strip().split(',')[-1])

    bundle_rewards = defaultdict(lambda : 0)
    bundles = defaultdict(lambda: [])
    involved_bundles = set()

    for block in data:
        if block['block_number'] != block_number:
            continue
        for tx in block["transactions"]:
            bundles[tx['bundle_index']].append(tx)

        for idx in bundles:
            for tx in bundles[idx]:
                bundle_rewards[idx] += int(tx['total_miner_reward'])
        
        for idx in bundles:
            for tx in bundles[idx]:
                if tx['transaction_hash'] in problem_transactions:
                    involved_bundles.add(idx)
        
        total_reward = 0
        for idx in involved_bundles:
            total_reward += bundle_rewards[idx]
        # print(int(block['miner_reward']))
        # print(involved_bundles)
        return total_reward
    return -1

args = parser.parse_args()
f = open(args.data, 'r')
data = json.load(f)

print('blocknumber,fb_mev')
for filename in glob.glob(problemsdir):
    fb_value = get_baseline_problem(filename, data)
    if fb_value >= 0:
        print('{},{}'.format(filename, float(fb_value)/1e18))
        sys.stdout.flush()
