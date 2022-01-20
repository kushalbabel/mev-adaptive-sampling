import os
import argparse
import logging
import pickle
import re
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from file_utils import gather_results
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Optimization')

    #------------ Arguments for transactios
    parser.add_argument('-p', '--path', help="path to results")
    parser.add_argument('-t', '--testset', help="name of testset")
    parser.add_argument('--reorder', action='store_true', help='optimize reordering of transactions')
    args = parser.parse_args()  

    path_to_save = 'plots'
    path_to_results = os.path.join(args.path, args.testset)

    patterns = [
                # '10iter_20nsamples_0.2random_0.0parents_0.5p_swap',
                # '20iter_10nsamples_0.2random_0.0parents_0.5p_swap',
                # '20iter_10nsamples_1.0random_0.0parents',
                '30iter_20nsamples_0.2random_0.0parents_0.5p_swap',
                '30iter_20nsamples_0.2random_0.0parents_0.8p_swap',
                '30iter_20nsamples_1.0random_0.0parents_0.5p_swap',
                # '30iter_20nsamples_0.2random_0.2parents_0.5p_swap',
                # '50iter_50nsamples_0.2random_0.4local_0.4_cross',
                # '50iter_50nsamples_1.0random_0.0local_0.0_cross',
                ]
    
    best_scores_all = []
    x_axis = []
    for i, p in enumerate(patterns):
        best_scores_all.append(gather_results(path_to_results, pattern=p))
        print(f'found {len(best_scores_all[-1].keys())} results with pattern {p}')

        n_iter = int(re.search('([0-9]+)iter', p).group(1))
        nsamples = int(re.search('_([0-9]+)nsamples', p).group(1))
        x_axis.append([(iter+1)*nsamples for iter in range(n_iter)])

        if '1.0random' in p:
            idx_random = i
    
    p = '30iter_20nsamples_0.2random_0.0parents_0.8p_swap'
    best_scores_all.append(gather_results('artifacts2/tests', pattern=p))
    patterns.append(p+'_0.1p_min')
    n_iter = int(re.search('([0-9]+)iter', p).group(1))
    nsamples = int(re.search('_([0-9]+)nsamples', p).group(1))
    x_axis.append([(iter+1)*nsamples for iter in range(n_iter)])
    print(f'found {len(best_scores_all[-1].keys())} results with pattern {patterns[-1]}')

    # p = '30iter_20nsamples_0.2random_0.0parents_1.0p_swap'
    # best_scores_all.append(gather_results('artifacts_adjacent_subset/tests', pattern=p))
    # patterns.append(p+'_0.1p_min_adjacent')
    # n_iter = int(re.search('([0-9]+)iter', p).group(1))
    # nsamples = int(re.search('_([0-9]+)nsamples', p).group(1))
    # x_axis.append([(iter+1)*nsamples for iter in range(n_iter)])
    # print(f'found {len(best_scores_all[-1].keys())} results with pattern {patterns[-1]}')

    #------ plot histogram of maximum MEV values found
    for i, s_dict in enumerate(best_scores_all):
        s_list = []
        for v in s_dict.values():
            assert v[-1]==np.max(v)
            s_list .append(v[-1])
        plt.clf()
        plt.hist(s_list, bins=50)
        plt.xlabel('maximum MEV')
        plt.ylabel('problem count')
        plt.savefig(os.path.join(path_to_save, 'MEVhist_{}_{}_{}.png'.format(patterns[i], args.testset, 'reorder' if args.reorder else 'alpha' )))

    #------ optionally remove some experiments with lower transaction number
    # best_scores = []
    # for i, s in enumerate(best_scores_all):
    #     scores_to_keep = {}
    #     problem_names = s.keys()
    #     for k in problem_names:
    #         problem = os.path.join('/home/kb742/mev-adaptive-sampling', args.testset, k)
    #         transactions_f = open(problem, 'r')
    #         transactions = transactions_f.readlines()
    #         length = len(transactions)
    #         if length >= 10:
    #             scores_to_keep[k] = s[k]
    #     print(f'kept {len(scores_to_keep.keys())} problems from pattern {patterns[i]}')
    #     best_scores.append(scores_to_keep)
    best_scores = best_scores_all

    #------ find common experiment names
    common_keys = list(best_scores[0].keys())
    for i in range(1, len(best_scores)):
        common_keys = np.intersect1d(common_keys, list(best_scores[i].keys()))
    
    #------ plot the percentage mev per sample count plots
    status = []
    lengths = []
    count = 0
    for k in common_keys:
        s = best_scores[0][k]
        try:
            max_score = np.max(np.concatenate([best_scores[i][k] for i in range(len(best_scores))], axis=0))
            max_score_nonrandom = np.max(np.concatenate([best_scores[i][k] for i in range(len(best_scores)) if i!=idx_random], axis=0))
            # if max_score != max_score_nonrandom:
            #     continue
            count += 1
            problem = os.path.join('/home/kb742/mev-adaptive-sampling', args.testset, k)
            transactions_f = open(problem, 'r')
            transactions = transactions_f.readlines()
            lengths.append(len(transactions))

        except:
            print(f'problem {k} did not exist in all optimization logs')
            continue
    
        n_iter = int(re.search('([0-9]+)iter', patterns[0]).group(1))
        s = np.expand_dims(np.pad(s, (0, n_iter-len(s)), mode='edge')/max_score, axis=0)
        try:
            status[0] = np.concatenate((status[0], s), axis=0)
        except:
            status.append(s)

        for i in range(1, len(best_scores)):
            n_iter = int(re.search('([0-9]+)iter', patterns[i]).group(1))
            s_ = np.expand_dims(np.pad(best_scores[i][k], (0, n_iter-len(best_scores[i][k])), mode='edge')/max_score, axis=0)
            try:
                status[i] = np.concatenate((status[i], s_), axis=0)
            except:
                status.append(s_)

    print(f'found {count} problems where adaptive sampling works better')
    print(lengths)

    plt.clf()
    for i in range(len(best_scores)):
        plt.plot(x_axis[i], np.mean(status[i], axis=0)*100., label=patterns[i])
    
    plt.legend()
    plt.xlabel('sample count')
    plt.ylabel('mean % of maximum MEV')
    plt.savefig(os.path.join(path_to_save, 'score_{}_{}.png'.format(args.testset, 'reorder' if args.reorder else 'alpha' )))

