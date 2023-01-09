import numpy as np
import os
import pickle
import yaml
import shutil

import sys
sys.path.append('backend/')

from simulate_client import simulate, setup


save_path = 'artifacts_smooth_uniswapv2'
if 'uniswapv2' in save_path:
    tx_path = '/home/kb742/mev-adaptive-sampling/eth_token_tests_uniswapv2/'
elif 'sushiswap' in save_path:
    tx_path = '/home/kb742/mev-adaptive-sampling/eth_token_tests/'
else:
    assert False
opt_pattern = '5iter_10nsamples_0.2random_0.0parents_0.1-0.8p_swap_neighbor'
zero_alphas = False


all_results = {}
eth_pairs = [f for f in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, f))]
for eth_pair in eth_pairs:
    if zero_alphas:
        path_to_yaml = os.path.join(save_path, eth_pair, 'noinsertion_mev_0alphas.yaml')
    else:
        path_to_yaml = os.path.join(save_path, eth_pair, 'noinsertion_mev.yaml')
    path_to_problems = [f for f in os.listdir(os.path.join(save_path, eth_pair)) if os.path.isdir(os.path.join(save_path, eth_pair,f))]
    noinsertion_mev_dict = {}
    for p in path_to_problems:
        problem = os.path.join(save_path, eth_pair, p)
        print(f'-------------{problem}')
        TRANSACTIONS = os.path.join(problem, opt_pattern, 'transactions')
        if not os.path.exists(TRANSACTIONS):
            shutil.copy(os.path.join(tx_path, eth_pair, p, 'amm_reduced'), TRANSACTIONS)
        with open(os.path.join(problem, opt_pattern, 'transactions'), 'r') as transactions_f:
            transaction_lines = transactions_f.readlines()
        new_transaction_lines = []
        for idx, line in enumerate(transaction_lines):
            elements = line.strip().split(',')
            tx_user = elements[1]
            if tx_user == 'miner':
                if zero_alphas:
                    for i, e in enumerate(elements):
                        if 'alpha' in e:
                            elements[i] = '0.0'
                    new_transaction_lines.append(','.join(elements))
                else:
                    continue
            else:
                new_transaction_lines.append(line)
        setup(new_transaction_lines[0])
        mev = simulate(new_transaction_lines, port_id=0, best=False, logfile=None, settlement='max')
        noinsertion_mev_dict[p] = mev
        all_results[f'{eth_pair}/{p}'] = mev
        
        with open(os.path.join(problem, opt_pattern, 'history_info.pkl'), 'rb') as f:
            logs = pickle.load(f)
        print('=> new MEV is {} and old MEV was {}'.format(mev, logs['best_scores'][-1]))
    
    with open(path_to_yaml, 'w') as f:
        yaml.dump(noinsertion_mev_dict, f)

fname = 'info_summary_noinsertion_0alphas.yaml' if zero_alphas else 'info_summary_noinsertion.yaml'
with open(os.path.join(save_path, fname), 'w') as f:
    yaml.dump(all_results, f)
