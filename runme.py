import os

path_to_problems = '/home/kb742/mev-adaptive-sampling/eth_token_tests/'
for eth_pair in os.listdir(path_to_problems): 
    # if eth_pair == '0x795065dcc9f64b5614c407a6efdc400da6221fb0':
    #     continue

    TRANSACTIONS = os.path.join(path_to_problems, eth_pair)
    DOMAIN = os.path.join(path_to_problems, eth_pair, 'domain')
    command = f'python optimize.py -t {TRANSACTIONS} -d {DOMAIN} --reorder --n_iter 5 --num_samples 10 --parents_portion 0.0 --p_swap_max 0.8 --p_swap_min 0.1 --num_samples_gauss 44'

    os.system(command)
