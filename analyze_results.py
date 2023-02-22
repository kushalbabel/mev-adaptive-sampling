import os
import math
import argparse
import pandas as pd
import re
import yaml
import pickle
import datetime
import matplotlib
from tqdm import tqdm
from collections import OrderedDict
from matplotlib import collections
from hashlib import new
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from file_utils import gather_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Optimization')

    #------------ Arguments for transactios
    parser.add_argument('-p', '--path', help="path to results")
    parser.add_argument('-s', '--section', help="which section of the lanturn does the plot belong to")
    args = parser.parse_args()  

    block_reward = 4.
    sorted_blocknum = True # if True, sorts by the block number, else by the mev
    path_to_save = 'plots/paper'
    os.makedirs(path_to_save, exist_ok=True)

    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14)

    if args.section == '5.3.1':
        #### load flashbots data
        # path_to_flashbots = '/home/kb742/mev-adaptive-sampling/data/flashbots_baseline.csv'
        # path_to_flashbots = 'flashbots_baseline_for_problems.csv'   # fair baseline
        path_to_flashbots = 'flashbots_baseline_for_uniswapv3_problems.csv'
        flashbots_data = pd.read_csv(path_to_flashbots)
        flashbots_logs_ = {str(flashbots_data['blocknumber'][idx]): flashbots_data['fb_mev'][idx] + block_reward for idx in range(len(flashbots_data.index))}
        if '/' in list(flashbots_logs_.keys())[0]:
            flashbots_logs = {}
            for k, v in flashbots_logs_.items():
                if ('sushiswap' in args.path and 'uniswap' in k) or \
                    ('sushiswap' not in args.path and not 'uniswap' in k):
                    continue
                new_k = k.split('/')[-2]
                if new_k in flashbots_logs:
                    flashbots_logs[new_k] = max(flashbots_logs[new_k], v)
                else:
                    flashbots_logs[new_k] = v
        else:
            flashbots_logs = flashbots_logs_

        #### load Lanturn data  
        path_to_results_yaml = os.path.join(args.path, 'info_summary.yaml')
        with open(path_to_results_yaml, 'r') as f:
            curr_results = yaml.safe_load(f)
        lanturn_results = {}
        for k in curr_results.keys():
            eth_pair, block_num = k.split('/')[0], k.split('/')[1]
            
            if block_num in lanturn_results:
                lanturn_results[block_num] = max(lanturn_results[block_num], curr_results[k])
            else:
                lanturn_results[block_num] = curr_results[k]  
        print(f'found {len(lanturn_results.keys())} Lantern results')

        common_keys = list(lanturn_results.keys())
        for k in common_keys:
            if not k in flashbots_logs:
                flashbots_logs[k] = block_reward

        #### remove flashbots problems with more transactions than lanturn
        if 'sushiswap' in args.path:
            contract_name = 'sushiswap'
        elif 'uniswapv2' in args.path:
            contract_name = 'uniswapv2'
        else:
            contract_name = 'uniswapv3'
        path_to_invalid_problems = f'analysis/{contract_name}_negatives'
        with open(path_to_invalid_problems, 'r') as f:
            invalid_problems = f.readlines()
        keys_to_remove = []
        count_removed = 0
        orig_count = len(common_keys)
        for l in invalid_problems:
            try:
                p_name = l.split('/')[-2]
            except:
                p_name = ''
            if p_name in common_keys:
                flasbots_mev = flashbots_logs[p_name]
                our_mev = lanturn_results[p_name]
                if flasbots_mev > our_mev:
                    common_keys.remove(p_name)
                    print(f'removing {p_name}')
                    count_removed += 1
        print(f'{count_removed}/{orig_count} problems removed')

        #### plot mev versus block number
        common_keys = [int(k) for k in common_keys]
        if sorted_blocknum:
            common_keys_sorted = np.sort(common_keys)
            # # plot block number versus date
            # df = pd.read_csv('/data/latest-data/block_times.csv')
            # dates = []
            # for k in common_keys_sorted:
            #     row_idx = df.index[df['block']==k]
            #     date = datetime.datetime.fromtimestamp(df['timestamp'][row_idx])
            #     dates.append(f'{date.year}/{date.month}/{date.day}')
            # print(dates)

        else:
            lanturn_values = [lanturn_results[str(common_keys[i])] for i in range(len(common_keys))]
            indices_sorted = np.argsort(lanturn_values)
            common_keys_sorted = np.asarray(common_keys)[indices_sorted]
        
        flashbots_values_sorted = np.asarray([flashbots_logs[str(common_keys_sorted[i])] for i in range(len(common_keys_sorted))])
        lanturn_values_sorted =  np.asarray([lanturn_results[str(common_keys_sorted[i])] for i in range(len(common_keys_sorted))])
        indices_to_keep = [i for i in range(len(common_keys_sorted)) if lanturn_values_sorted[i] >= block_reward + 1.]    

        plt.clf()
        fig, ax = plt.subplots(figsize=(12, 4))
        plt.plot(lanturn_values_sorted[indices_to_keep], label='Lanturn')
        plt.plot(flashbots_values_sorted[indices_to_keep], label='Flashbots')
        plt.legend(fontsize=16)
        plt.xlabel('Block number', fontsize=16)
        ax.set_xticks(range(len(indices_to_keep)))
        xticklabels = []
        for i, idx in enumerate(indices_to_keep):
            xticklabels.append(common_keys_sorted[idx] if i%2==0 else '')
        ax.set_xticklabels(xticklabels, rotation=60)
        plt.ylabel('MEV', fontsize=16)
        plt.yscale('log')
        fname = '{}.png'.format(args.path.replace('artifacts_smooth_', ''))
        plt.savefig(os.path.join(path_to_save, fname), bbox_inches='tight')

    elif args.section == '5.3.2':
        #### load combo data  
        path_to_combo = 'artifacts_smooth_combo'
        with open(os.path.join(path_to_combo, 'info_summary.yaml'), 'r') as f:
            curr_results = yaml.safe_load(f)
        combo_results = {}
        for k in curr_results.keys():
            eth_pair, block_num = k.split('/')[0], k.split('/')[1]
            
            if block_num in combo_results:
                combo_results[block_num] = max(combo_results[block_num], curr_results[k])
            else:
                combo_results[block_num] = curr_results[k]  
        print(f'found {len(combo_results.keys())} combo results')

        #### load Lanturn data  
        path_to_results_yaml = os.path.join(args.path, 'info_summary.yaml')
        with open(path_to_results_yaml, 'r') as f:
            curr_results = yaml.safe_load(f)
        lanturn_results = {}
        for k in curr_results.keys():
            eth_pair, block_num = k.split('/')[0], k.split('/')[1]
            
            if block_num in lanturn_results:
                lanturn_results[block_num] = max(lanturn_results[block_num], curr_results[k])
            else:
                lanturn_results[block_num] = curr_results[k]  
        print(f'found {len(lanturn_results.keys())} Lantern results')

        common_keys = np.intersect1d(list(lanturn_results.keys()), list(combo_results.keys()))

        #### plot mev versus block number
        common_keys = [int(k) for k in common_keys]
        if sorted_blocknum:
            common_keys_sorted = np.sort(common_keys)
        else:
            lanturn_values = [lanturn_results[str(common_keys[i])] for i in range(len(common_keys))]
            indices_sorted = np.argsort(lanturn_values)
            common_keys_sorted = np.asarray(common_keys)[indices_sorted]
        
        combo_values_sorted = np.asarray([combo_results[str(common_keys_sorted[i])] for i in range(len(common_keys_sorted))])
        lanturn_values_sorted =  np.asarray([lanturn_results[str(common_keys_sorted[i])] for i in range(len(common_keys_sorted))])
        indices_to_keep = [i for i in range(len(common_keys_sorted)) if combo_values_sorted[i] > lanturn_values_sorted[i]+0.1] #range(len(common_keys))#[i for i in range(len(common_keys_sorted)) if combo_values_sorted[i] < 200]    

        plt.clf()
        fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7,4))
        # ax.set_yscale('log'), ax2.set_yscale('log')
        ax.plot(lanturn_values_sorted[indices_to_keep], label='Sushiswap')
        ax.plot(combo_values_sorted[indices_to_keep], label='Sushiswap + UniswapV2')
        ax2.plot(lanturn_values_sorted[indices_to_keep], label='Sushiswap')
        ax2.plot(combo_values_sorted[indices_to_keep], label='Sushiswap + UniswapV2')

        ax2.set_ylim(0, 40)
        ax.set_ylim(270, 530)

        # hide the spines between ax and ax2
        ax.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.yaxis.tick_left()
        ax.tick_params(bottom=False, labelbottom=False)
        ax2.yaxis.tick_left()

        d = .015 # how big to make the diagonal lines in axes coordinates
        # arguments to pass plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d,+d), (-d-0.01,+d-0.01), **kwargs)
        ax2.plot((-d,+d),(-0.07-d,-0.07+d), **kwargs)
        # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        # ax2.plot((1-d,1+d), (-d,+d), **kwargs)
        # ax2.plot((-d,+d), (-d,+d), **kwargs)

        ax.legend(fontsize=16)
        ax2.set_xlabel('Block number', fontsize=16)
        ax2.set_xticks(range(len(indices_to_keep)))
        xticklabels = []
        for i, idx in enumerate(indices_to_keep):
            xticklabels.append(common_keys_sorted[idx] if i%1==0 else '')
        ax2.set_xticklabels(xticklabels, rotation=45)
        fig.supylabel('MEV', fontsize=16)
        # plt.yscale('log')
        fig.subplots_adjust(hspace=0.1)
        fname = '{}_combo.png'.format(args.path.replace('artifacts_smooth_', ''))
        plt.savefig(os.path.join(path_to_save, fname), bbox_inches='tight')

    elif args.section == '5.7':
        opt_pattern =  '5iter_10nsamples_0.2random_0.0parents_0.1-0.8p_swap_neighbor'
        x_axis = 'time' # or 'n_samples'

        with open(os.path.join(args.path, 'info_summary.yaml'), 'r') as f:
            mevs = yaml.safe_load(f)
        with open(os.path.join(args.path, 'times_summary.yaml'), 'r') as f:
            times = yaml.safe_load(f)

        plt.figure()
        if x_axis == 'time':
            t_max = 3600 # 1 hour
            t_unit = 0.1
            fig = plt.figure(figsize=(5,3))
            steps = np.arange(t_unit, t_max, t_unit)
            percentages = {i:[] for i in steps}
            for k, mev in mevs.items():
                if mev < 4.5:
                    continue
                with open(os.path.join(args.path, k, opt_pattern, 'history_info.pkl'), 'rb') as f:
                    logs = pickle.load(f)
                time_periter = [logs['all_subsamples'][i]['n_subsamples']/44 * times[k] for i in range(len(logs['all_subsamples']))]
                scores_periter = logs['all_scores']*100./np.max(logs['best_scores'])

                moving_percentage = [scores_periter[0]]
                moving_time = [time_periter[0]]
                for t, score in zip(time_periter[1:], scores_periter[1:]):
                    moving_percentage.append(max(moving_percentage[-1], score))
                    moving_time.append(moving_time[-1] + t)
                moving_percentage = [0] + moving_percentage
                moving_time = [0] + moving_time
                # plt.plot(moving_time, moving_percentage)

                i = 0
                for s in steps:
                    if s <= np.max(moving_time):
                        if s >= moving_time[i+1]:
                            i += 1
                        percentages[s].append(moving_percentage[i])
                    else:
                        percentages[s].append(100.)

        else:  # plot score progress versus number of samples
            n_max_samples = 50000
            fig = plt.figure(figsize=(5,4))
            steps = np.arange(1, n_max_samples)
            percentages = {i:[] for i in steps}
            for k, mev in mevs.items():
                if mev < 4.5:
                    continue
                with open(os.path.join(args.path, k, opt_pattern, 'history_info.pkl'), 'rb') as f:
                    logs = pickle.load(f)
                n_simulations_periter = [logs['all_subsamples'][i]['n_subsamples'] for i in range(len(logs['all_subsamples']))]
                scores_periter = logs['all_scores']*100./np.max(logs['best_scores'])

                moving_percentage = [scores_periter[0]]
                moving_sample_count = [n_simulations_periter[0]]
                for n_samples, score in zip(n_simulations_periter[1:], scores_periter[1:]):
                    moving_percentage.append(max(moving_percentage[-1], score))
                    moving_sample_count.append(moving_sample_count[-1] + n_samples)
                moving_percentage = [0] + moving_percentage
                moving_sample_count = [0] + moving_sample_count
                # plt.plot(moving_sample_count, moving_percentage)

                i = 0
                for s in steps:
                    if s <= np.max(moving_sample_count):
                        if s == moving_sample_count[i+1]:
                            i += 1
                        percentages[s].append(moving_percentage[i])
                    else:
                        percentages[s].append(100.)

        if x_axis == 'time': plt.xscale('log')
        plt.plot(steps, [np.min(percentages[s]) for s in steps], label='Min')
        plt.plot(steps, [np.quantile(percentages[s], q=0.25) for s in steps], label='Q1')
        plt.plot(steps, [np.median(percentages[s]) for s in steps], label='Median')
        plt.plot(steps, [np.quantile(percentages[s], q=0.75) for s in steps], label='Q3')
        plt.plot(steps, [np.max(percentages[s]) for s in steps], label='Max')
        plt.legend()
        plt.ylabel('MEV Percentile', fontsize=16), plt.xlabel('Time (s)', fontsize=16)
        plt.savefig(os.path.join(path_to_save, '{}_score_progress_vs_{}.png'.format(args.path.replace('artifacts_smooth_', ''), 'time' if x_axis=='time' else 'nsamples')), bbox_inches='tight')

    elif args.section == 'random':
        rand_pattern =  '50iter_44nsamples_1.0random_0.0local_0.0_cross'
        ours_pattern = '50iter_44nsamples_0.2random_0.4local_0.4_cross'
        x_axis = 'n_samples' # or 'time'

        with open(os.path.join(args.path, 'info_summary_alphas_random.yaml'), 'r') as f:
            mevs_random = yaml.safe_load(f)
        with open(os.path.join(args.path, 'info_summary_alphas_adaptive.yaml'), 'r') as f:
            mevs_ours = yaml.safe_load(f)
        with open(os.path.join(args.path, 'times_summary.yaml'), 'r') as f:
            times = yaml.safe_load(f)

        plt.figure()
        if x_axis == 'time':
            t_max = 300
            t_unit = 0.1
            fig = plt.figure(figsize=(5,3))
            steps = np.arange(t_unit, t_max, t_unit)
            percentages_random = {i:[] for i in steps}
            percentages_ours = {i:[] for i in steps}
            for k, mev in mevs_random.items():
                if not k in times:
                    continue

                with open(os.path.join(args.path, k, rand_pattern, 'history_info_0.pkl'), 'rb') as f:
                    logs_random = pickle.load(f)
                with open(os.path.join(args.path, k, ours_pattern, 'history_info_0.pkl'), 'rb') as f:
                    logs_ours = pickle.load(f)

                time_periter_random = len(logs_random['all_samples'])/44 * times[k]
                time_periter_ours = len(logs_ours['all_samples'])/44 * times[k]
                scores_periter_random = [np.max(logs_random['all_scores'][i:i+44]) for i in range(math.ceil(len(logs_random['all_scores'])/44))]
                scores_periter_ours = [np.max(logs_ours['all_scores'][i:i+44]) for i in range(math.ceil(len(logs_ours['all_scores'])/44))]

                moving_mev_random = [scores_periter_random[0]]
                moving_time_random = [time_periter_random]
                for score in scores_periter_random[1:]:
                    moving_mev_random.append(max(moving_mev_random[-1], score))
                    moving_time_random.append(moving_time_random[-1] + time_periter_random)
                moving_mev_random = [0] + moving_mev_random
                moving_time_random = [0] + moving_time_random

                print(moving_mev_random)
                print(moving_time_random)

                moving_mev_ours = [scores_periter_ours[0]]
                moving_time_ours = [time_periter_ours]
                for score in scores_periter_ours[1:]:
                    moving_mev_ours.append(max(moving_mev_ours[-1], score))
                    moving_time_ours.append(moving_time_ours[-1] + time_periter_ours)
                moving_mev_ours = [0] + moving_mev_ours
                moving_time_ours = [0] + moving_time_ours

                i = 0
                for s in steps:
                    if s <= np.max(moving_time_random):
                        if s >= moving_time_random[i+1]:
                            i += 1
                        percentages_random[s].append(moving_mev_random[i])
                        print(len(percentages_random[s]))
                    else:
                        percentages_random[s].append(percentages_random[s][-1])
                i = 0
                for s in steps:
                    if s <= np.max(moving_time_ours):
                        if s >= moving_time_ours[i+1]:
                            i += 1
                        percentages_ours[s].append(moving_mev_ours[i])
                    else:
                        percentages_ours[s].append(percentages_ours[s][-1])

                exit()

        else:  # plot score progress versus number of samples
            n_max_samples = 50 * 44
            fig = plt.figure(figsize=(5,3))
            steps = np.arange(n_max_samples)
            percentages_random = {i:[] for i in steps}
            percentages_ours = {i:[] for i in steps}
            for k in mevs_random.keys():
                if not k in times:
                    continue
                with open(os.path.join(args.path, k, rand_pattern, 'history_info_0.pkl'), 'rb') as f:
                    logs_random = pickle.load(f)
                with open(os.path.join(args.path, k, ours_pattern, 'history_info_0.pkl'), 'rb') as f:
                    logs_ours = pickle.load(f)

                moving_score_random = [logs_random['all_scores'][0]]
                moving_score_ours = [logs_ours['all_scores'][0]]
                for s_r in logs_random['all_scores'][1:]:
                    moving_score_random.append(max(moving_score_random[-1], s_r))
                for s_o in logs_ours['all_scores'][1:]:
                    moving_score_ours.append(max(moving_score_ours[-1], s_o))
                moving_score_random = [0] + moving_score_random
                moving_score_ours = [0] + moving_score_ours

                for i, s in enumerate(steps):
                    if s <= len(logs_random['all_samples']):
                        percentages_random[s].append(moving_score_random[i])
                    else:
                        percentages_random[s].append(np.max(logs_random['all_scores']))

                for i, s in enumerate(steps):
                    if s <= len(logs_ours['all_samples']):
                        percentages_ours[s].append(moving_score_ours[i])
                    else:
                        percentages_ours[s].append(np.max(logs_ours['all_scores']))

        if x_axis == 'time': plt.xscale('log')
        plt.plot(steps, [np.max(percentages_random[s]) for s in steps], label='Random')
        plt.plot(steps, [np.max(percentages_ours[s]) for s in steps], label='Lanturn')
        plt.legend()
        plt.ylabel('MEV', fontsize=16), plt.xlabel('# samples', fontsize=16)
        plt.savefig(os.path.join(path_to_save, 'random_vs_lanturn_{}.png'.format('time' if x_axis=='time' else 'nsamples')), bbox_inches='tight')


'''        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Optimization')

    #------------ Arguments for transactios
    parser.add_argument('-p', '--path', help="path to results")
    parser.add_argument('-t', '--testset', default=None, help="name of testset")
    parser.add_argument('--reorder', action='store_true', help='optimize reordering of transactions')
    parser.add_argument('--flashbots', action='store_true', help='compare MEVs with flashbots data')
    parser.add_argument('--SA', action='store_true', help='compare MEVs with SA reordering')
    args = parser.parse_args()  

    block_reward = 4.
    path_to_save = os.path.join('plots', args.testset) if args.testset is not None else 'plots'
    os.makedirs(path_to_save, exist_ok=True)
    path_to_results = args.path

    patterns = [
                '5iter_10nsamples_0.2random_0.0parents_0.1-0.8p_swap_neighbor',
                # '5iter_10nsamples_0.2random_0.0parents_0.1-0.8p_swap_neighbor_old',
                # '5iter_10nsamples_1.0random_0.0parents_0.1-0.8p_swap_neighbor',
                # '50iter_44nsamples_0.2random_0.4local_0.4_cross',
                # '50iter_44nsamples_1.0random_0.0local_0.0_cross',
                ]

    if args.SA:
        path_to_SA = 'artifacts_smooth_sushiswap_SA' #args.path + '_SA'
        patterns.append('SA_50iter_1nsamples')

    flashbots_data = None
    if args.flashbots:
        # path_to_flashbots = '/home/kb742/mev-adaptive-sampling/data/flashbots_baseline.csv'
        path_to_flashbots = 'flashbots_baseline_for_problems.csv'   #fair baseline
        flashbots_data = pd.read_csv(path_to_flashbots)
        patterns.append('flashbots')
    
    noinsertion_mev_dict = None
    # load mev baseline with no miner insertions
    # eth_pairs = [p for p in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, p))]
    # noinsertion_mev_dict = {}
    # summary_dict = {}
    # for eth_pair in eth_pairs:
    #     path_to_yaml = os.path.join(args.path, eth_pair, 'noinsertion_mev_0alphas.yaml')
    #     with open(path_to_yaml, 'r') as f:
    #         mev_info = yaml.safe_load(f)
    #     for k, v in mev_info.items():
    #         if k in noinsertion_mev_dict.keys():
    #             noinsertion_mev_dict[k] = max(v, noinsertion_mev_dict[k])
    #         else:
    #             noinsertion_mev_dict[k] = v

    #         summary_dict[f'{eth_pair}/{k}'] = v
    # # print(summary_dict)
    # # with open(os.path.join(args.path, 'info_summary_noinsertion.yaml'), 'w') as f:
    # #     yaml.dump(summary_dict, f)

    best_scores_all = []
    x_axis = []
    idx_random = None
    for i, p in enumerate(patterns):
        if 'flashbots' in p:
            flashbots_idx = i
            flashbots_logs_ = {str(flashbots_data['blocknumber'][idx]): flashbots_data['fb_mev'][idx] + block_reward for idx in range(len(flashbots_data.index))}
            if '/' in list(flashbots_logs_.keys())[0]:
                flashbots_logs = {}
                for k, v in flashbots_logs_.items():
                    if ('sushiswap' in args.path and 'uniswapv2' in k) or \
                        ('sushiswap' not in args.path and not 'uniswapv2' in k):
                        continue
                    new_k = k.split('/')[-2]
                    if new_k in flashbots_logs:
                        flashbots_logs[new_k] = max(flashbots_logs[new_k], v)
                    else:
                        flashbots_logs[new_k] = v
            else:
                flashbots_logs = flashbots_logs_
            best_scores_all.append(flashbots_logs)
        else:
            if 'old' in p:
                curr_results, _ = gather_results('./artifacts_smooth_oldMEV', pattern='5iter_10nsamples_0.2random_0.0parents_0.1-0.8p_swap_neighbor')
                best_scores_all.append(curr_results)
            else:
                if 'SA' in p:
                    path_to_results = path_to_SA
                    SA_idx = i
                else:
                    ours_idx = i
                curr_results, eth_pairs = gather_results(path_to_results, pattern=p)
                best_scores_all.append(curr_results)
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
    common_keys = np.asarray(list(best_scores[0].keys()))
    for i in range(2, len(best_scores)):
        common_keys = np.intersect1d(common_keys, list(best_scores[i].keys()))
    common_keys = common_keys.tolist()
    print('problem names:', common_keys)
    if 'sushiswap' in args.path:
        common_keys.remove('13118320') # problems in sushiswap that ruin the MEV plot scale
    #     common_keys.remove('13450883')
    #     common_keys.remove('13323340')
    for k in common_keys:
        if not k in best_scores[flashbots_idx].keys():
            best_scores[flashbots_idx][k] = [block_reward]

    dict_to_save = {}
    for k in common_keys:
        dict_to_save[f'{eth_pairs[k]}/{k}'] = np.max(best_scores[0][k]).tolist()
    with open(os.path.join(args.path, 'info_summary.yaml'), 'w') as f:
        yaml.dump(dict_to_save, f)

    #------ plot histogram of maximum MEV values found
    for i, s_dict in enumerate(best_scores_all):
        s_list = []
        for k, v in s_dict.items():
            if not k in common_keys:
                continue
            if isinstance(v, list):
                if np.isnan(v[-1]):
                    continue 
                s_list.append(v[-1])
                assert v[-1]==np.max(v)
            else:
                s_list .append(v)
        plt.figure()
        plt.hist(s_list, bins=20)
        plt.xlabel('maximum MEV')
        plt.ylabel('problem count')
        plt.savefig(os.path.join(path_to_save, 'MEVhist_{}_{}.png'.format(patterns[i], 'reorder' if args.reorder else 'alpha' )))
    
    #------- list of problems to remove
    assert 'sushiswap' in args.path or 'uniswapv2' in args.path, 'contract name (sushiswap or uniswapv2) must be in the path argument'
    path_to_invalid_problems = 'analysis/{}_negatives'.format('sushiswap' if 'sushiswap' in args.path else 'uniswapv2')
    with open(path_to_invalid_problems, 'r') as f:
        invalid_problems = f.readlines()
    keys_to_remove = []
    count_removed = 0
    orig_count = len(common_keys)
    for l in invalid_problems:
        try:
            p_name = l.split('/')[-2]
        except:
            p_name = ''
        if p_name in common_keys:
            flasbots_mev = np.max(best_scores[flashbots_idx][p_name])
            our_mev = np.max(best_scores[ours_idx][p_name])
            if flasbots_mev > our_mev:
                common_keys.remove(p_name)
                print(f'removing {p_name}')
                count_removed += 1
    print(f'{count_removed}/{orig_count} problems removed')
    
    #------- plot mev versus problem number
    common_keys = np.asarray(common_keys)
    ours_main = np.asarray([np.max(best_scores[0][k]) for k in common_keys])
    indices_to_keep = (ours_main <= np.max(ours_main))
    sort_indices = np.argsort(ours_main[indices_to_keep]).tolist()
    plt.clf()
    fig, ax = plt.subplots(figsize=(20, 6))
    for i in range(len(best_scores)):
        curr_data = np.asarray([np.max(best_scores[i][k]) for k in common_keys])[indices_to_keep][sort_indices]
        plt.plot(curr_data, label=patterns[i])
    if noinsertion_mev_dict is not None:
        plt.plot(np.asarray([noinsertion_mev_dict[k] for k in common_keys])[indices_to_keep][sort_indices], label='no miner insertions')
    plt.legend()
    plt.xlabel('problem number')
    ax.set_xticks(range(len(common_keys[indices_to_keep])))
    ax.set_xticklabels(np.asarray(common_keys)[indices_to_keep][sort_indices], rotation=90)
    plt.ylabel('MEV')
    plt.yscale('log')
    fname = 'fairbaseline_mev_{}'.format('reorder' if args.reorder else 'alpha')
    fname += '_SA.png' if args.SA else '.png'
    plt.savefig(os.path.join(path_to_save, fname), bbox_inches='tight')

    if args.flashbots:
        best_scores.pop(-1) #remove flashbots logs
    
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

        for i in range(1, len(best_scores)):
            n_iter = int(re.search('([0-9]+)iter', patterns[i]).group(1))
            s_ = np.expand_dims(np.pad(best_scores[i][k], (0, n_iter-len(best_scores[i][k])), mode='edge')/max_score, axis=0)
            try:
                status[i] = np.concatenate((status[i], s_), axis=0)
            except:
                status.append(s_)

    print(f'found {count} problems where adaptive sampling works better')

    if args.flashbots:
        best_scores.pop(-1) #remove flashbots logs
    plt.clf()
    for i in range(len(best_scores)):
        plt.plot(x_axis[i], np.mean(status[i], axis=0)*100., label=patterns[i])
    
    plt.legend()
    plt.xlabel('sample count')
    plt.ylabel('mean % of maximum MEV')
    plt.savefig(os.path.join(path_to_save, 'score_{}.png'.format('reorder' if args.reorder else 'alpha' )))

'''
# path_to_artifacts = 'artifacts_smooth_sushiswap'
# opt_pattern =  '5iter_10nsamples_0.2random_0.0parents_0.1-0.8p_swap_neighbor'

# with open(os.path.join(path_to_artifacts, 'info_summary.yaml'), 'r') as f:
#         mevs = yaml.safe_load(f)
# for k, v in mevs.items():
#     if mev < 4.5:
#         continue

#     with open(os.path.join(path_to_artifacts, k, opt_pattern, 'history_info.pkl'), 'rb') as f:
#         logs = pickle.load(f)

#     n_simulations_periter = [logs['all_subsamples'][i]['n_subsamples'] for i in range(len(logs['all_subsamples']))]
    # scores_periter = logs['best_scores']*100./np.max(logs['best_scores'])
