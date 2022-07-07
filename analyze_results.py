import os
import argparse
import logging
import pickle
import pandas as pd
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
    parser.add_argument('-t', '--testset', default=None, help="name of testset")
    parser.add_argument('--reorder', action='store_true', help='optimize reordering of transactions')
    parser.add_argument('--flashbots', action='store_true', help='compare MEVs with flashbots data')
    args = parser.parse_args()  

    block_reward = 4.
    path_to_save = os.path.join('plots', args.testset)
    os.makedirs(path_to_save, exist_ok=True)
    path_to_results = args.path
    # if args.testset is not None:
    #     path_to_results = os.path.join(path_to_results, args.testset)

    patterns = [
                '5iter_10nsamples_0.2random_0.0parents_0.1-0.8p_swap_neighbor',
                '5iter_10nsamples_1.0random_0.0parents_0.1-0.8p_swap_neighbor'
                ]
    
    flashbots_data = None
    if args.flashbots:
        path_to_flashbots = '/home/kb742/mev-adaptive-sampling/flashbots_baseline.csv'
        flashbots_data = pd.read_csv(path_to_flashbots)
        patterns.append('flashbots')
    
    best_scores_all = []
    x_axis = []
    for i, p in enumerate(patterns):
        if 'flashbots' in p:
            best_scores_all.append({str(flashbots_data['blocknumber'][idx]): flashbots_data['fb_mev'][idx] + block_reward for idx in range(len(flashbots_data.index))})
        else:
            best_scores_all.append(gather_results(path_to_results, pattern=p))
            n_iter = int(re.search('([0-9]+)iter', p).group(1))
            nsamples = int(re.search('_([0-9]+)nsamples', p).group(1))
            x_axis.append([(iter+1)*nsamples for iter in range(n_iter)])

        print(f'found {len(best_scores_all[-1].keys())} results with pattern {p}')
        
        if '1.0random' in p:
            idx_random = i

    #------ plot histogram of maximum MEV values found
    for i, s_dict in enumerate(best_scores_all):
        s_list = []
        for v in s_dict.values():
            if isinstance(v, list):
                s_list.append(v[-1])
                assert v[-1]==np.max(v)
            else:
                s_list .append(v)
        plt.clf()
        plt.hist(s_list, bins=50)
        plt.xlabel('maximum MEV')
        plt.ylabel('problem count')
        plt.savefig(os.path.join(path_to_save, 'MEVhist_{}_{}.png'.format(patterns[i], 'reorder' if args.reorder else 'alpha' )))

    
    #------ optionally remove some experiments with lower transaction number
    best_scores = best_scores_all
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
    

    #------ find common experiment names
    common_keys = list(best_scores[0].keys())
    for i in range(1, len(best_scores)):
        common_keys = np.intersect1d(common_keys, list(best_scores[i].keys()))
    print('problem names:', common_keys)

    #------- plot mev versus problem number
    ours_main = [np.max(best_scores[0][k]) for k in common_keys]
    sort_indices = np.argsort(ours_main).tolist()
    plt.clf()
    fig, ax = plt.subplots()
    for i in range(len(best_scores)):
        curr_data = np.asarray([np.max(best_scores[i][k]) for k in common_keys])[sort_indices]
        plt.plot(curr_data, label=patterns[i])
    
    plt.legend()
    plt.xlabel('problem number')
    ax.set_xticks(range(len(common_keys)))
    ax.set_xticklabels(common_keys[sort_indices], rotation=45)
    plt.ylabel('MEV')
    plt.savefig(os.path.join(path_to_save, 'baseline_mev_{}.png'.format('reorder' if args.reorder else 'alpha')), bbox_inches='tight')

    best_scores.pop(-1) #remove flashbots logs
    #------ plot the percentage mev per sample count plots
    status = []
    lengths = []
    count = 0
    for k in common_keys:
        s = best_scores[0][k]

        max_score = np.max(np.concatenate([best_scores[i][k] for i in range(len(best_scores))], axis=0))
        max_score_nonrandom = np.max(np.concatenate([best_scores[i][k] for i in range(len(best_scores)) if i!=idx_random], axis=0))
        # if max_score != max_score_nonrandom:
        #     continue
        if max_score == max_score_nonrandom:
            count += 1
        problem = os.path.join('/home/kb742/mev-adaptive-sampling/eth_token_tests', args.testset, k, 'amm_reduced')
        transactions_f = open(problem, 'r')
        transactions = transactions_f.readlines()
        lengths.append(len(transactions))
    
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

    plt.clf()
    for i in range(len(best_scores)):
        plt.plot(x_axis[i], np.mean(status[i], axis=0)*100., label=patterns[i])
    
    plt.legend()
    plt.xlabel('sample count')
    plt.ylabel('mean % of maximum MEV')
    plt.savefig(os.path.join(path_to_save, 'score_{}.png'.format('reorder' if args.reorder else 'alpha' )))

