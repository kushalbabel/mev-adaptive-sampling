import os
import yaml
import numpy as np

from util import uniswap_to_sushiswap, sushiswap_to_uniswap, sushiswap_to_uniswapv3, uniswapv2_to_uniswapv3


# optimizing only selected problems
with open('artifacts_smooth_combo_v2v3/info_summary.yaml', 'r') as f:
    old_results = yaml.safe_load(f)
keys_list = list(old_results.keys())
values_list = [old_results[k] for k in keys_list]
sorted_idx = np.argsort(values_list)

# begin_idx = 0
# end_idx = len(sorted_idx)-1
# sorted_values = []
# sorted_keys = []
# for i in range(len(values_list)):
#     if i%2 == 0:
#         sorted_values.append(values_list[sorted_idx[begin_idx]])
#         sorted_keys.append(keys_list[sorted_idx[begin_idx]])
#         begin_idx += 1
#         assert begin_idx <= (end_idx + 1)
#     else:
#         sorted_values.append(values_list[sorted_idx[end_idx]])
#         sorted_keys.append(keys_list[sorted_idx[end_idx]])
#         end_idx -= 1
#         assert begin_idx <= (end_idx + 1)

sorted_values = np.asarray(values_list)[sorted_idx][::-1]
sorted_keys = np.asarray(keys_list)[sorted_idx][::-1]
for k in sorted_keys:
    eth_pair, block_num = k.split('/')[0], k.split('/')[1]
    new_k = uniswapv2_to_uniswapv3(eth_pair)
    # new_k = sushiswap_to_uniswap(eth_pair)
    
    TRANSACTION = os.path.join('/home/kb742/mev-adaptive-sampling/eth_token_tests_uniswapv3', new_k, block_num, 'amm_reduced')
    DOMAIN = os.path.join('/home/kb742/mev-adaptive-sampling/eth_token_tests_uniswapv3', new_k, 'domain')
    DEXES = 'uniswapv3'
    
    if os.path.isfile(TRANSACTION):
        command = f'python optimize.py -t {TRANSACTION} -d {DOMAIN} --dexes {DEXES} --reorder --n_iter 5 --num_samples 10 --parents_portion 0.0 --p_swap_max 0.8 --p_swap_min 0.1 --num_samples_gauss 40 --n_parallel_gauss 40'
        os.system(command)


'''
path_to_problems = '/home/kb742/mev-adaptive-sampling/eth_token_tests_uniswap_composition'
# path_to_problems = '/home/kb742/mev-adaptive-sampling/eth_token_tests_uniswapv3'
# path_to_problems = '/home/kb742/mev-adaptive-sampling/eth_token_tests_v2composition'
# path_to_problems = '/home/kb742/mev-adaptive-sampling/eth_token_tests_uniswapv2'
# path_to_problems = '/home/kb742/mev-adaptive-sampling/eth_token_tests/'
# path_to_ignored_problems = 'analysis/{}_negatives'.format('uniswapv2') #'sushiswap'
for eth_pair in os.listdir(path_to_problems): 
    
    TRANSACTIONS = os.path.join(path_to_problems, eth_pair)
    DOMAIN = os.path.join(path_to_problems, eth_pair, 'domain')
    DEXES = 'uniswapv2 uniswapv3'

    if os.path.exists(DOMAIN):
        command = f'python optimize.py -t {TRANSACTIONS} -d {DOMAIN} --dexes {DEXES} --reorder --n_iter 5 --num_samples 10 --parents_portion 0.0 --p_swap_max 0.8 --p_swap_min 0.1 --num_samples_gauss 40 --n_parallel_gauss 40'
        # command = f'python optimize.py -t {TRANSACTIONS} -d {DOMAIN} --ignore {path_to_ignored_problems} --reorder --n_iter 5 --num_samples 10 --parents_portion 0.0 --p_swap_max 0.8 --p_swap_min 0.1 --num_samples_gauss 44'
        # command = f'python optimize.py -t {TRANSACTIONS} -d {DOMAIN} --reorder --SA --n_iter 50 --num_samples 1'

        os.system(command)
'''
