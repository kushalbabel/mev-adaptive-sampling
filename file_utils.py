import os
import shutil
import pickle
import re
import csv
import yaml
import time
import numpy as np
import multiprocessing as mp

from simulate_client import simulate, setup

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


def gather_results(path, pattern):
    paths = gather_result_paths(path, pattern, fname='history_info.pkl')

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
    def __init__(self, transactions):
        self.transactions = transactions
        setup(transactions[0])
    
    def evaluate(self, port_id):
        t0 = time.time()
        mev = simulate(self.transactions, port_id)
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
    path_to_results = 'artifacts_smooth'
    pattern = '5iter_10nsamples_0.2random_0.0parents_0.1-0.8p_swap_neighbor' #'50iter_44nsamples_0.2random_0.4local_0.4_cross' #'5iter_10nsamples_0.2random_0.0parents_0.1-0.8p_swap_neighbor' #'10iter_15nsamples_0.2random_0.0parents_0.1-0.8p_swap_neighbor'
    curr_results, eth_pairs = gather_results(path_to_results, pattern=pattern)
    summary_dict = {}
    for k, v in curr_results.items():
        eth_pair = eth_pairs[k]
        summary_dict[f'{eth_pair}/{k}'] = v[-1]
    print(summary_dict)
    with open(os.path.join(path_to_results, 'info_summary.yaml'), 'a') as f:
        yaml.dump(summary_dict, f)


    # ##### code snippet for saving per-sample simulation time
    # path_to_results = 'artifacts_smooth'
    # pattern = '5iter_10nsamples_0.2random_0.0parents_0.1-0.8p_swap_neighbor'
    # n_repeat = 44

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
        
    #     timer_obj = Timer(transactions)
        
    #     curr_times = []
    #     new_mevs = []
    #     for _ in range(5):
    #         with mp.Pool() as e:
    #             batch_output = list(e.map(timer_obj.evaluate, range(n_repeat)))
    #         new_mevs += [batch_output[i][0] for i in range(len(batch_output))]
    #         curr_times += [batch_output[i][1] for i in range(len(batch_output))]
    #     assert new_mevs[0] == orig_mevs[k]
    #     times_summary[k] = np.mean(curr_times).tolist()
    #     print(k, times_summary[k])

    # with open(os.path.join(path_to_results, 'times_summary.yaml'), 'w') as f:
    #     yaml.dump(times_summary, f)


