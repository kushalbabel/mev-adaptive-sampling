import numpy as np
import os
import pickle
import shutil

from optimize import reorder

save_path = 'artifacts_smooth_sushiswap'
opt_pattern = '5iter_10nsamples_0.2random_0.0parents_0.1-0.8p_swap_neighbor'

eth_pairs = os.listdir(save_path)
for eth_pair in eth_pairs:
    path_to_problems = [f.path for f in os.scandir(os.path.join(save_path, eth_pair)) if f.is_dir()]
    for problem in path_to_problems:
        print(f'-------------{problem}')
        TRANSACTIONS = os.path.join(problem, opt_pattern, 'transactions')
        DOMAIN = os.path.join('/home/kb742/mev-adaptive-sampling/eth_token_tests', eth_pair,'domain')
        # if os.path.exists(TRANSACTIONS) and os.path.exists(os.path.join(problem, '50iter_44nsamples_1.0random_0.0local_0.0_cross', 'history_info_0.pkl')):
        #     print('====== already optimized')
        #     continue
        try:
            with open(os.path.join(problem, opt_pattern, 'transactions'), 'r') as transactions_f:
                transaction_lines = transactions_f.readlines()
        except:
            problem_name = problem.split('/')[-1]
            print('=> copying from', os.path.join('artifacts_smooth_oldMEV', eth_pair, problem_name, opt_pattern, 'transactions'), 
                            'to', os.path.join(problem, opt_pattern))
            shutil.copy(os.path.join('artifacts_smooth_oldMEV', eth_pair, problem_name, opt_pattern, 'transactions'), os.path.join(problem, opt_pattern))
            with open(os.path.join(problem, opt_pattern, 'transactions'), 'r') as transactions_f:
                transaction_lines = transactions_f.readlines()

        # print('original transactions:', transaction_lines)
        transactions_dict = {}
        miner_idx = 0
        for idx, line in enumerate(transaction_lines[1:]):
            if line.startswith('#'):
                continue
            elements = line.strip().split(',')
            tx_user = elements[1]
            if tx_user == 'miner':
                user_id = f'M{str(miner_idx)}'
                assert not user_id in transactions_dict
                transactions_dict[user_id] = [idx]
                miner_idx += 1
            else:
                user_id = elements[1]
                if user_id in transactions_dict:
                    transactions_dict[user_id].append(idx)
                else:
                    transactions_dict[user_id] = [idx]

        with open(os.path.join(problem, opt_pattern, 'history_info.pkl'), 'rb') as f:
            logs = pickle.load(f)
        sample = logs['best_samples'][-1]
        for i, s in enumerate(sample):
            sample[i] = transactions_dict[s].pop(0)
        for k, v in transactions_dict.items():
            assert len(v)==0, f'{k}, {v}'

        best_order = reorder(transaction_lines, sample)
        # print('best order:', best_order)
        
        
        with open(TRANSACTIONS, 'w') as flog:
            for tx in best_order:
                flog.write('{}\n'.format(tx.strip()))
        # command = f'python optimize.py -t {TRANSACTIONS} -d {DOMAIN} --num_samples_gauss 44'
        command = f'python optimize.py -t {TRANSACTIONS} -d {DOMAIN} --u_random_portion_gauss 1.0 --local_portion 0.0 --cross_portion 0.0 --num_samples_gauss 44'
        os.system(command)
        
        print('=> old MEV was {}'.format(logs['best_scores'][-1]))
        exit()