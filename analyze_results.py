import os
import argparse
import logging
import pickle
import re
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from simulate import simulate
from sampling_utils import Gaussian_sampler, RandomOrder_sampler

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
        problem_name = re.search('(problem_[0-9]+)', p).group(1)
        results[problem_name] = info['best_scores']

    return results
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Optimization')

    #------------ Arguments for transactios
    parser.add_argument('-p', '--path', help="path to results")
    parser.add_argument('-t', '--testset', help="name of testset")
    parser.add_argument('--reorder', action='store_true', help='optimize reordering of transactions')
    
    args = parser.parse_args()  
    
    # with open('artifacts/tests/problem_12400000/30iter_20nsamples_1.0random_0.0parents_0.5p_swap/history_info.pkl', 'rb') as f:
    #     info = pickle.load(f)
    # scores = info['all_scores']
    # best_scores = info['best_scores']
    # plt.hist(scores, bins=100)
    # plt.savefig('hist.png')
    # plt.clf()
    # plt.plot(best_scores)
    # plt.savefig('best_scores.png')
    # exit()

    patterns = [
                # '10iter_20nsamples_0.2random_0.0parents_0.5p_swap',
                # '20iter_10nsamples_0.2random_0.0parents_0.5p_swap',
                # '20iter_10nsamples_0.2random_0.0parents_0.8p_swap',
                # '20iter_10nsamples_0.2random_0.0parents_0.3p_swap',
                # '20iter_10nsamples_1.0random_0.0parents',
                '30iter_20nsamples_1.0random_0.0parents',
                '30iter_20nsamples_0.2random_0.0parents_0.5p_swap'
                ]
    path_to_results = os.path.join(args.path, args.testset)

    scores = []
    x_axis = []
    for p in patterns:
        scores.append(gather_results(path_to_results, pattern=p))
        print(f'found {len(scores[-1].keys())} results with pattern {p}')

        n_iter = int(re.search('([0-9]+)iter', p).group(1))
        nsamples = int(re.search('_([0-9]+)nsamples', p).group(1))
        x_axis.append([(iter+1)*nsamples for iter in range(n_iter)])

    #------ plot histogram of maximum MEV values found
    for i, s_dict in enumerate(scores):
        s_list = []
        for v in s_dict.values():
            assert v[-1]==np.max(v)
            s_list .append(v[-1])
        plt.clf()
        plt.hist(s_list, bins=50)
        plt.xlabel('maximum MEV')
        plt.ylabel('problem count')
        plt.savefig('MEVhist_{}_{}_{}.png'.format(patterns[i], args.testset, 'reorder' if args.reorder else 'alpha' ))

    #------ finding common experiment names
    common_keys = list(scores[0].keys())
    for i in range(1, len(scores)):
        common_keys = np.intersect1d(common_keys, list(scores[i].keys()))
    
    #------ plotting the percentage mev per sample count plots
    status = []
    for k in common_keys:
        s = scores[0][k]
        try:
            max_score = np.max(np.concatenate([scores[i][k] for i in range(len(scores))], axis=0))
        except:
            print(f'problem {k} did not exist in all optimization logs')
            continue
    
        n_iter = int(re.search('([0-9]+)iter', patterns[0]).group(1))
        s = np.expand_dims(np.pad(s, (0, n_iter-len(s)), mode='edge')/max_score, axis=0)
        try:
            status[0] = np.concatenate((status[0], s), axis=0)
        except:
            status.append(s)

        for i in range(1, len(scores)):
            n_iter = int(re.search('([0-9]+)iter', patterns[i]).group(1))
            s_ = np.expand_dims(np.pad(scores[i][k], (0, n_iter-len(scores[i][k])), mode='edge')/max_score, axis=0)
            try:
                status[i] = np.concatenate((status[i], s_), axis=0)
            except:
                status.append(s_)

    plt.clf()
    for i in range(len(scores)):
        plt.plot(x_axis[i], np.mean(status[i], axis=0)*100., label=patterns[i])
    
    plt.legend()
    plt.xlabel('sample count')
    plt.ylabel('mean % of maximum MEV')
    plt.savefig(os.path.join('score_{}_{}.png'.format(args.testset, 'reorder' if args.reorder else 'alpha' )))

