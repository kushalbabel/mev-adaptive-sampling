import os
import shutil
import pickle
import re
import csv
import yaml
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import repeat

# from simulate_client import simulate, setup

def copy_files(file_patterns, orig_path='./artifacts_', path_to_move='./artifacts'):
    for problem in os.listdir(orig_path):
        subfolders = os.listdir(os.path.join(orig_path, problem))
        for sf in subfolders:
            if sf in file_patterns:
                curr_path = os.path.join(orig_path, problem, sf)
                dest_path = os.path.join(path_to_move, problem, sf)
                print(f'copying {curr_path} to {dest_path}')
                shutil.copytree(curr_path, dest_path, dirs_exist_ok=True)

def delete_files(file_pattern, path='./artifacts'):
    for problem in os.listdir(path):
        subfolders = os.listdir(os.path.join(path, problem))
        for sf in subfolders:
            if sf == file_pattern:
                curr_path = os.path.join(path, problem, sf)
                print(f'removing {curr_path}')
                shutil.rmtree(curr_path)

def rename_files(file_pattern, new_name, path='./artifacts'):
    for problem in os.listdir(path):
        subfolders = os.listdir(os.path.join(path, problem))
        for sf in subfolders:
            if sf == file_pattern:
                curr_path = os.path.join(path, problem, sf)
                print(f'renaming {curr_path}')
                os.rename(curr_path, os.path.join(path, problem, new_name))
                
def count_transactions(path_to_testset):
    def gather_result_paths(path):
        paths = []
        if os.path.isfile(path):
                return [path]
        else:
            for d in os.listdir(path):
                paths += gather_result_paths(os.path.join(path, d))
        return paths

    paths = gather_result_paths(path_to_testset)
    lengths = []
    for p in paths:
        transactions_f = open(p, 'r')
        transactions = transactions_f.readlines()
        lengths.append(len(transactions)-1)
    
    print('minimum transaction count:', np.min(lengths))
    print('maximum transaction count:', np.max(lengths))
    print('median transaction count:', np.median(lengths))
    print('# random samples based on median:', np.math.factorial(int(np.max(lengths)))/1000)


def gather_result_paths(path, pattern, fname):
    paths = []
    if os.path.isdir(path):
        if pattern in path:
            f = os.path.join(path, fname)
            if os.path.exists(f):
                return [f]
            else:
                print(f"no {fname} found in {path}")
        else:
            for d in os.listdir(path):
                paths += gather_result_paths(os.path.join(path, d), pattern, fname)
    return paths


def gather_results(path, pattern, fname='history_info.pkl'):
    paths = gather_result_paths(path, pattern, fname=fname)

    results, eth_pairs = {}, {}
    for p in paths:
        with open(p, 'rb') as f:
            info = pickle.load(f)
        eth_pair_idx = re.search('(0x[a-z0-9]+)', p).span()[1]
        eth_pair = re.search('(0x[a-z0-9]+)', p).group(1)
        problem_name = p[eth_pair_idx:].split('/')[1]
        if problem_name in results.keys():
            print(f'============== found duplicate problem name {problem_name}')
            if np.max(info['best_scores']) > np.max(results[problem_name]):
                results[problem_name] = info['best_scores'].tolist()
                print('==== replacing')
                eth_pairs[problem_name] = eth_pair
        else:
            results[problem_name] = info['best_scores'].tolist()
            eth_pairs[problem_name] = eth_pair

    return results, eth_pairs

def dump_csvs(path, testset, pattern):
    best_scores = gather_results(os.path.join(path, testset), pattern)
    n_iter = int(re.search('([0-9]+)iter', pattern).group(1))
    rows = [['transaction'] + [f'iter_{i}' for i in range(1, n_iter+1)]]
    for k, v in best_scores.items():
        rows.append([k] + v.tolist())
    with open(f'info_{testset}_{pattern}.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


class Timer():
    def __init__(self, transactions, involved_dexes, n_repeat=-1):
        self.transactions = transactions
        self.involved_dexes = involved_dexes

        ctx = simulate_efficient_hardhat.setup(self.transactions[0])
        if n_repeat > 1:
            with mp.Pool() as e:
                self.ctx_list = list(e.starmap(simulate_efficient_hardhat.prepare_once, zip(repeat(ctx), repeat(self.transactions), range(n_repeat), repeat(self.involved_dexes))))
        else:
            self.ctx_list = [simulate_efficient_hardhat.prepare_once(ctx, self.transactions, 0, self.involved_dexes)]
     
    def evaluate(self, port_id):
        t0 = time.time()
        mev = simulate_efficient_hardhat.simulate(self.ctx_list[port_id], self.transactions, port_id, involved_dexes=self.involved_dexes, best=False, logfile=None, settlement='max')
        # mev = simulate(self.transactions, port_id)
        t1 = time.time() - t0

        return mev, t1


if __name__ == '__main__':
    # dump_csvs('./artifacts_earlystopping', testset='tests', pattern='50iter_50nsamples_0.2random_0.4local_0.4_cross')
    
    # file_patterns = ['50iter_50nsamples_0.2random_0.4local_0.4_cross',
    #                  '50iter_50nsamples_1.0random_0.0local_0.0_cross']
    # copy_files(file_patterns, orig_path='./artifacts_/tests', path_to_move='./artifacts/tests')
    
    # rename_files(file_pattern='30iter_20nsamples_0.2random_0.0parents_1.0p_swap', 
    #                 new_name='30iter_20nsamples_0.2random_0.0parents_0.1-1.0p_swap_adjsubset', 
    #                 path='artifacts_adjacent_subset/tests')

    # count_transactions(path_to_testset='/home/kb742/mev-adaptive-sampling/tests/')

    ##### code snippet for restructuring the clienttest files to match the sampling code
    # PATH = '/home/gid-javaheripim/clienttests_sushi_'
    # new_PATH = '/home/gid-javaheripim/clienttests_sushi'
    # os.makedirs(new_PATH, exist_ok=True)
    # all_files = [os.path.join(dp, f) for dp, _, filenames in os.walk(PATH) for f in filenames]
    # for f in all_files:
    #     problem_type = os.path.basename(f)
    #     problem_number = os.path.basename(os.path.dirname(f))
    #     new_name = f'problem_{problem_number}' + ('_reduced' if 'reduced' in problem_type else '')
        
    #     new_f = os.path.join(new_PATH, new_name)
    #     shutil.copyfile(f, new_f)
    #     print(f'moving {f} to {new_f}')


    ##### code snippet for merging logs from two directories into one of them
    # src = 'artifacts_smooth_uniswapv2_updatedTX_over14e6'
    # dst = 'artifacts_smooth_uniswapv2_updatedTX'

    # eth_pairs_src = [p for p in os.listdir(src) if os.path.isdir(os.path.join(src, p))]
    # eth_pairs_dst = [p for p in os.listdir(dst) if os.path.isdir(os.path.join(dst, p))]
    # for eth_pair in eth_pairs_src:
    #     curr_path = os.path.join(src, eth_pair)
    #     if eth_pair not in eth_pairs_dst:
    #         print(f'moving {curr_path} to {dst}')
    #         shutil.copytree(curr_path, os.path.join(dst, eth_pair), dirs_exist_ok=True)
    #     else:
    #         problem_names = [p for p in os.listdir(curr_path) if os.path.isdir(os.path.join(curr_path, p))]
    #         for p_name in problem_names:
    #             print(f'moving {os.path.join(curr_path, p_name)} to {os.path.join(dst, eth_pair)}')
    #             assert not os.path.exists(os.path.join(dst, eth_pair, p_name))
    #             shutil.copytree(os.path.join(curr_path, p_name), os.path.join(dst, eth_pair, p_name), dirs_exist_ok=True)

    ##### code snippet for saving info_summary files
    path_to_results = 'artifacts_smooth_aave_oraclerelated'
    pattern = '40iter_400nsamples_0.4random_0.3local_0.3_cross' #'5iter_10nsamples_0.2random_0.0parents_0.1-0.8p_swap_neighbor'
    curr_results, eth_pairs = gather_results(path_to_results, pattern=pattern, fname='history_info_0.pkl')
    summary_dict = {}
    for k, v in curr_results.items():
        eth_pair = eth_pairs[k]
        summary_dict[f'{eth_pair}/{k}'] = v[-1]
    print(summary_dict)
    with open(os.path.join(path_to_results, 'info_summary.yaml'), 'w') as f:
        yaml.dump(summary_dict, f)

    ##### code snippet for saving per-sample simulation time
    # import sys
    # sys.path.append('backend/')
    # import simulate_efficient_hardhat

    # path_to_results = 'artifacts_smooth_sushiswap'
    # pattern = '5iter_10nsamples_0.2random_0.0parents_0.1-0.8p_swap_neighbor'
    # n_repeat = 1 #5

    # with open(os.path.join(path_to_results, 'info_summary.yaml'), 'r') as f:
    #     orig_mevs = yaml.safe_load(f)

    # paths = gather_result_paths(path_to_results, pattern, fname='transactions_optimized')
    # times_summary = {}
    # for p in paths:
    #     with open(p, 'r') as transactions_f:
    #         transactions = transactions_f.readlines()
    #     eth_pair_idx = re.search('(0x[a-z0-9]+)', p).span()[1]
    #     eth_pair = re.search('(0x[a-z0-9]+)', p).group(1)
    #     problem_name = p[eth_pair_idx:].split('/')[1]
    #     k = f'{eth_pair}/{problem_name}'

    #     timer_obj = Timer(transactions, involved_dexes=['sushiswap'], n_repeat=n_repeat)
        
    #     curr_times = []
    #     new_mevs = []
    #     for _ in range(5):
    #         if n_repeat > 1:
    #             with mp.Pool() as e:
    #                 batch_output = list(e.map(timer_obj.evaluate, range(n_repeat)))
    #         else:
    #             batch_output = [timer_obj.evaluate(port_id=0)]
    #         new_mevs += [batch_output[i][0] for i in range(len(batch_output))]
    #         curr_times += [batch_output[i][1] for i in range(len(batch_output))]
    #     # assert new_mevs[0] == orig_mevs[k], f'new mev:{new_mevs[0]}, orig mev:{orig_mevs[k]}'
    #     if new_mevs[0] is None:
    #         print('MEV IS NONE')
    #         print(f'{k} -> new mev:{new_mevs[0]}, orig mev:{orig_mevs[k]}')
    #     else:
    #         if abs(new_mevs[0] - orig_mevs[k]) > 0.5:
    #             print(f'{k} -> new mev:{new_mevs[0]}, orig mev:{orig_mevs[k]}')
    #     times_summary[k] = np.mean(curr_times).tolist()
        # print(k, times_summary[k])

    # with open(os.path.join(path_to_results, f'times_summary_nparallel_{n_repeat}_new.yaml'), 'w') as f:
    #     yaml.dump(times_summary, f)
    

    # n_parallels = [5, 10, 15, 20, 25, 40]
    # times = {}
    # for n_p in n_parallels:
    #     with open(os.path.join(path_to_results, f'times_summary_nparallel_{n_p}_new.yaml'), 'r') as f: #f'times_summary_nparallel_{n_repeat}.yaml'), 'r') as f:
    #         times[n_p] = yaml.safe_load(f)

    # keys = list(times[n_parallels[0]].keys())
    # avg_speedup = {}
    # for n_p in n_parallels[1:]:
    #     avg_speedup[f'{n_p}/{n_parallels[0]}'] = np.mean([times[n_parallels[0]][k]/times[n_p][k] for k in keys]) * n_p/n_parallels[0]
    # print(avg_speedup)

    # for k in times.keys():
    #     with open(os.path.join(path_to_results, k, pattern, 'history_info.pkl'), 'rb') as f:
    #         logs = pickle.load(f)
    #     time_periter = [logs['all_subsamples'][i]['n_subsamples']/n_repeat * times[k] for i in range(len(logs['all_subsamples']))]
    #     scores_periter = logs['all_scores']*100./np.max(logs['best_scores'])

    #     print(k, len(time_periter))
    #     total_time = np.sum(time_periter)

    ##### code snippet for comparing with baseline
    path_to_results = 'artifacts_smooth_aave_oraclerelated'
    baseline_df = pd.read_csv('/home/kb742/mev-adaptive-sampling/data/flashbots_baseline_for_aave.csv', sep=',', header=None)
    block_reward = 4.

    with open(os.path.join(path_to_results, 'info_summary.yaml'), 'r') as f:
        our_mevs = yaml.safe_load(f)

    # os.makedirs('artifacts_smooth_aave_outperforming', exist_ok=True)
    count, total = 0, 0
    doing_worse = {}
    for path, mev in zip(baseline_df.iloc[1:, 0], baseline_df.iloc[1:, 1]):
        transaction_id = path.split('/')[-3] + '/' + path.split('/')[-2]

        if transaction_id in our_mevs:
            our_mev = our_mevs[transaction_id]
            total += 1

            if our_mev >= float(mev) + block_reward:
                print(transaction_id, our_mev, float(mev) + block_reward)
                count += 1
                # if not os.path.exists(os.path.join('artifacts_smooth_aave_outperforming', transaction_id)):
                #     shutil.copytree(os.path.join(path_to_results, transaction_id), os.path.join('artifacts_smooth_aave_outperforming', transaction_id))
            else:
                doing_worse[path] = {'our_mev': our_mev, 'baseline': float(mev) + block_reward}
    
    print(f'outperformed on {count}/{total} problems')
    # with open(os.path.join(path_to_results, 'aave_needs_reordering.yaml'), 'w') as f:
    #     yaml.dump(doing_worse, f)






    # from optimize import substitute
    # from simulate_efficient_hardhat import simulate, setup

    # def get_params(transactions):
    #     params = list()
    #     for transaction in transactions:
    #         vals = transaction.split(',')
    #         for val in vals:
    #             if 'alpha' in val:
    #                 if val.strip() not in params:
    #                     params.append(val.strip())
    #     return list(params)

    # transactions_f = open('/home/gid-javaheripim/mev-adaptive-sampling/artifacts_smooth_aave_oracle/0xdeadc0de/14745388/50iter_400nsamples_0.4random_0.3local_0.3_cross/transactions', 'r')
    # transactions = transactions_f.readlines()
    # domain = '/home/kb742/mev-adaptive-sampling/tests_liquidations_oracle/0xdeadc0de/14745388/domain'
    # domain_f = open(domain, 'r')
    # domain_scales = {}
    # for line in domain_f.readlines():
    #     if line[0] == '#':
    #         continue
    #     tokens = line.strip().split(',')

    #     # TODO: add other currencies here
    #     lower_lim, upper_lim = float(tokens[1]), float(tokens[2])
    #     token_pair = domain.split('/')[-2]

    #     if len(tokens)==3:
    #         VALID_RANGE[token_pair] = min(1e6, upper_lim)
    #         if upper_lim > VALID_RANGE[token_pair]:
    #             domain_scales[tokens[0]] = upper_lim / VALID_RANGE[token_pair]
    #             upper_lim = VALID_RANGE[token_pair]
    #         else:
    #             domain_scales[tokens[0]] = 1.0
    #     else:
    #         assert len(tokens)==4
    #         domain_scales[tokens[0]] = float(tokens[3])
    # print('domain scales:', domain_scales)

    # with open('/home/gid-javaheripim/mev-adaptive-sampling/artifacts_smooth_aave_oracle/0xdeadc0de/14745388/50iter_400nsamples_0.4random_0.3local_0.3_cross/history_info_0.pkl', 'rb') as f:
    #     logs = pickle.load(f)
    # id_best_overall = np.argmax(logs['best_scores'])
    # best_sample_overall = logs['best_samples'][id_best_overall]
    # best_sample = logs['best_samples'][-1]
    # best_score = logs['best_scores'][-1]
    # params = get_params(transactions)

    # sample_dict = {}
    # for p_name, v in zip(params, best_sample):
    #     print(f'{p_name}: {v * domain_scales[p_name]}')
    #     sample_dict[p_name] = v * domain_scales[p_name]
    # datum = substitute(transactions, sample_dict, cast_to_int=True)
    # print(datum)

    # ctx = setup(transactions[0], capital=10000)
    # mev = simulate(ctx, datum, 0, involved_dexes=['sushiswap', 'aave', 'uniswapv3'])
    # print(mev, best_score)
    



