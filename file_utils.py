import os
import shutil
import pickle
import re
import csv
import numpy as np

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

def gather_results(path, pattern):
    def gather_result_paths(path, pattern):
        paths = []
        if os.path.isdir(path):
            if pattern in path:
                f = os.path.join(path, 'history_info.pkl')
                if os.path.exists(f):
                    return [f]
            else:
                for d in os.listdir(path):
                        paths += gather_result_paths(os.path.join(path, d), pattern)
        return paths

    paths = gather_result_paths(path, pattern)
    results = {}
    for p in paths:
        with open(p, 'rb') as f:
            info = pickle.load(f)
        problem_name = p.split('/')[-3] #re.search('(problem_[0-9]+)', p).group(1)
        results[problem_name] = info['best_scores']

    return results

def dump_csvs(path, testset, pattern):
    best_scores = gather_results(os.path.join(path, testset), pattern)
    n_iter = int(re.search('([0-9]+)iter', pattern).group(1))
    rows = [['transaction'] + [f'iter_{i}' for i in range(1, n_iter+1)]]
    for k, v in best_scores.items():
        rows.append([k] + v.tolist())
    with open(f'info_{testset}_{pattern}.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


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
    PATH = '/home/gid-javaheripim/clienttests_sushi_'
    new_PATH = '/home/gid-javaheripim/clienttests_sushi'
    os.makedirs(new_PATH, exist_ok=True)
    all_files = [os.path.join(dp, f) for dp, _, filenames in os.walk(PATH) for f in filenames]
    for f in all_files:
        problem_type = os.path.basename(f)
        problem_number = os.path.basename(os.path.dirname(f))
        new_name = f'problem_{problem_number}' + ('_reduced' if 'reduced' in problem_type else '')
        
        new_f = os.path.join(new_PATH, new_name)
        shutil.copyfile(f, new_f)
        print(f'moving {f} to {new_f}')
