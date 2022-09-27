import os
import argparse
from collections import OrderedDict
from matplotlib import collections
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
    path_to_save = os.path.join('plots', args.testset) if args.testset is not None else 'plots'
    os.makedirs(path_to_save, exist_ok=True)
    path_to_results = args.path

    patterns = [
                '5iter_10nsamples_0.2random_0.0parents_0.1-0.8p_swap_neighbor',
                # '5iter_10nsamples_1.0random_0.0parents_0.1-0.8p_swap_neighbor'
                ]

    flashbots_data = None
    if args.flashbots:
        path_to_flashbots = '/home/kb742/mev-adaptive-sampling/data/flashbots_baseline.csv'
        flashbots_data = pd.read_csv(path_to_flashbots)
        patterns.append('flashbots')
    
    best_scores_all = []
    x_axis = []
    idx_random = None
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
    common_keys = [str(i) for i in [13179381, 13525407, 13262636, 13420743, 13519520, 13539852, 13539877, 13501761, 13526069, 13406200, 13022130, 13529335, 13504619, 13450855, 13076476, 13285119, 13103508, 13235453, 13115319, 13534463, 13119385, 13450863, 13236404, 13457441, 13179367, 13142903, 13238591, 13119472, 13486527, 13121478, 13491910, 13389694, 13377448, 13450892, 13184922, 13536521, 13185021, 13118320, 13323340, 13359998, 13075089, 13450883]]
    # common_keys = list(best_scores[0].keys())
    for i in range(1, len(best_scores)):
        common_keys = np.intersect1d(common_keys, list(best_scores[i].keys()))
    print('problem names:', common_keys)

    #------ plot histogram of maximum MEV values found
    for i, s_dict in enumerate(best_scores_all):
        s_list = []
        for k, v in s_dict.items():
            if not k in common_keys:
                continue
            if isinstance(v, list):
                s_list.append(v[-1])
                assert v[-1]==np.max(v)
            else:
                s_list .append(v)
        plt.figure()
        plt.hist(s_list, bins=20)
        plt.xlabel('maximum MEV')
        plt.ylabel('problem count')
        plt.savefig(os.path.join(path_to_save, 'MEVhist_{}_{}.png'.format(patterns[i], 'reorder' if args.reorder else 'alpha' )))

    #------- plot mev versus problem number
    ours_main = [np.max(best_scores[0][k]) for k in common_keys]
    sort_indices = np.argsort(ours_main).tolist()
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(best_scores)):
        curr_data = np.asarray([np.max(best_scores[i][k]) for k in common_keys])[sort_indices]
        plt.plot(curr_data, label=patterns[i])
    
    plt.legend()
    plt.xlabel('problem number')
    ax.set_xticks(range(len(common_keys)))
    ax.set_xticklabels(np.asarray(common_keys)[sort_indices], rotation=90)
    plt.ylabel('MEV')
    plt.savefig(os.path.join(path_to_save, 'baseline_mev_{}.png'.format('reorder' if args.reorder else 'alpha')), bbox_inches='tight')

    # best_scores.pop(-1) #remove flashbots logs
    #------ plot the percentage mev per sample count plots
    status = []
    count = 0
    for k in common_keys:
        s = best_scores[0][k]

        curr_scores = []
        for i in range(len(best_scores)):
            v = best_scores[i][k]
            if isinstance(v, list):
                curr_scores += v
            else:
                curr_scores += [v]
            
        max_score = np.max(curr_scores)
        max_score_ours = np.max(best_scores[0][k])
        if max_score == max_score_ours:
            count += 1
        
        n_iter = int(re.search('([0-9]+)iter', patterns[0]).group(1))
        s = np.expand_dims(np.pad(s, (0, n_iter-len(s)), mode='edge')/max_score, axis=0)
        try:
            status[0] = np.concatenate((status[0], s), axis=0)
        except:
            status.append(s)

        for i in range(1, len(best_scores[:-1])):
            n_iter = int(re.search('([0-9]+)iter', patterns[i]).group(1))
            s_ = np.expand_dims(np.pad(best_scores[i][k], (0, n_iter-len(best_scores[i][k])), mode='edge')/max_score, axis=0)
            try:
                status[i] = np.concatenate((status[i], s_), axis=0)
            except:
                status.append(s_)

    print(f'found {count} problems where adaptive sampling works better')

    best_scores.pop(-1) #remove flashbots logs
    plt.clf()
    for i in range(len(best_scores)):
        plt.plot(x_axis[i], np.mean(status[i], axis=0)*100., label=patterns[i])
    
    plt.legend()
    plt.xlabel('sample count')
    plt.ylabel('mean % of maximum MEV')
    plt.savefig(os.path.join(path_to_save, 'score_{}.png'.format('reorder' if args.reorder else 'alpha' )))

