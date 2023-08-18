from builtins import hasattr
import os
import shutil
import argparse
import logging
import pickle
import copy
import math
import re
import yaml
from pathlib import Path
from tqdm import tqdm
from itertools import repeat
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import git

import sys
sys.path.append('backend/')

# from simulate import simulate
# from simulate_client import simulate, setup
from simulate_efficient_hardhat import simulate, setup, prepare_once
from sampling_utils import Gaussian_sampler, RandomOrder_sampler, SA_sampler



VALID_RANGE = {}

def get_params(transactions):
    params = list()
    for transaction in transactions:
        vals = transaction.split(',')
        for val in vals:
            if 'alpha' in val:
                if val.strip() not in params:
                    params.append(val.strip())
    return list(params)

def next_sample(params, domain):
    sample = {}
    for param in params:
        sample[param] = domain[param][0] # naively try the start of range
    return sample

def substitute(transactions, sample, cast_to_int=False):
    datum = []
    for transaction in transactions:
        transaction = transaction.strip()
        required_params = get_params([transaction])
        for param in required_params:
            if cast_to_int:
                transaction = transaction.replace(param, str(int(sample[param])))
            else:
                transaction = transaction.replace(param, str(sample[param]))
        datum.append(transaction)
    return datum

# transactions: parametric list of transactions (a transaction is csv values)
class MEV_evaluator(object):
    def __init__(self, transactions, params, domain_scales, involved_dexes, ctx_list, cast_to_int=False):
        self.transactions = transactions
        self.params = params
        self.domain_scales = domain_scales
        self.involved_dexes = involved_dexes
        self.ctx_list = ctx_list
        self.cast_to_int = cast_to_int
        
    def evaluate(self, sample, port_id, best=False, logfile=None):
        # sample is a vector that has the values for parameter names in self.params    
        sample_dict = {}
        for p_name, v in zip(self.params, sample):
            sample_dict[p_name] = v * self.domain_scales[p_name]
            # print(sample_dict)
        datum = substitute(self.transactions, sample_dict, cast_to_int=self.cast_to_int)
        logging.info(datum)

        mev = simulate(self.ctx_list[port_id], datum, port_id, involved_dexes=self.involved_dexes, best=best, logfile=logfile)

        if mev is None:
            os.makedirs('nan_problems', exist_ok=True)
            with open(os.path.join('nan_problems',f'sample_{np.random.randint(100)}'), 'w') as flog:
                for tx in datum:
                    flog.write('{}\n'.format(tx.strip()))

        return mev


def reorder(transactions, order):
    '''
        function to reorder a set of transactions, except for the first one
    '''
    order = order.astype(np.int32)
    reordered_transactions = [transactions[0]] + [transactions[i+1] for i in order]
    return reordered_transactions


def get_groundtruth_order(transaction_lines, include_miner=True):
    user_transactions = {}
    miner_idx = 0
    seen_mint_fees = {}  #for keeping track of mint and burnAndCollect txs
    for idx, line in enumerate(transaction_lines):
        if line.startswith('#'):
            #TODO remove for performance in prod
            continue
        elements = line.strip().split(',')
        tx_user = elements[1]
        if tx_user != 'miner':
            user_id = elements[1]
            if user_id in user_transactions:
                user_transactions[user_id].append(idx)
            else:
                user_transactions[user_id] = [idx]
        elif include_miner:
            if elements[2] != 'position_manager':
                user_id = 'M' + str(miner_idx)
                assert not user_id in user_transactions
                user_transactions[user_id] = [idx]
                miner_idx += 1
            else:
                # handle mint and burn and collect pairs
                if elements[7] not in seen_mint_fees:
                    user_id = 'M' + str(miner_idx)
                    assert not user_id in user_transactions
                    user_transactions[user_id] = [idx]
                    seen_mint_fees[elements[7]] = user_id
                    miner_idx += 1
                else:
                    user_id = seen_mint_fees[elements[7]]
                    if elements[4] == 'mint':
                        user_transactions[user_id] = [idx] + user_transactions[user_id]
                    elif elements[4] == 'burnAndCollect':
                        user_transactions[user_id] = user_transactions[user_id] + [idx]
                    else:
                        raise NotImplementedError

    return user_transactions


class Reorder_evaluator(object):
    def __init__(self, transactions, domain, domain_scales, n_iter_gauss, num_samples, minimum_num_good_samples, u_random_portion, local_portion, cross_portion, 
                pair_selection, alpha_max, early_stopping, save_path, n_parallel, involved_dexes, ctx_list, use_repr=False, groundtruth_order=None, cast_to_int=False):
        self.transactions = transactions
        self.domain = domain
        self.domain_scales = domain_scales

        # arguments for the gaussian sampler 
        self.n_iter_gauss = n_iter_gauss
        self.num_samples = num_samples
        self.minimum_num_good_samples = minimum_num_good_samples
        self.u_random_portion = u_random_portion
        self.local_portion = local_portion
        self.cross_portion = cross_portion
        self.pair_selection = pair_selection
        self.alpha_max = alpha_max
        self.early_stopping = early_stopping
        self.save_path = os.path.join(save_path, f'{n_iter_gauss}iter_{num_samples}nsamples_{u_random_portion}random_{local_portion}local_{cross_portion}_cross')
        self.n_parallel = n_parallel

        self.involved_dexes = involved_dexes
        self.ctx_list = ctx_list

        self.use_repr = use_repr
        self.groundtruth_order = groundtruth_order
        assert self.groundtruth_order is not None, 'need to provide the original order of transactions if using abstract representation'

        self.cast_to_int = cast_to_int # cast alpha values to integer before calling mev simulation
    
    def translate_sample(self, sample_repr):
        gt_order = copy.deepcopy(self.groundtruth_order)
        sample = []
        for s in sample_repr:
            sample.append(gt_order[s].pop(0))
        for v in gt_order.values():
            assert len(v)==0
    
        return np.asarray(sample)

    def check_constraints(self, sample):
        def check_order(array1, array2):
            if not isinstance(array2, list):
                array2 = array2.tolist()
            
            indices = np.asarray([])
            for element in array1:
                idx = array2.index(element)
                if not np.all(indices <= idx):
                    return False
                indices = np.append(indices, idx)
            return True
    
        for user_order in self.groundtruth_order.values():
            if len(user_order)==1:
                continue
            flag = check_order(user_order, sample)
            if not flag:
                return False

        return True
            
    def evaluate_randomAlphas(self, sample, rand_alphas, port_id=0, **kwargs):
        if self.use_repr:
            sample = self.translate_sample(sample)
            assert self.check_constraints(sample)
        
        transactions = reorder(self.transactions, sample)
        params = get_params(transactions)
        boundaries = []
        for p_name in params:
            boundaries.append(list(self.domain[p_name]))
        boundaries = np.asarray(boundaries)
        curr_rand_alphas = [rand_alphas[k] for k in params]
        # curr_rand_alphas = [np.random.uniform(boundaries[i][0], boundaries[i][1]) for i in range(len(boundaries))]

        mev_evaluator = MEV_evaluator(transactions, params, domain_scales=self.domain_scales, involved_dexes=self.involved_dexes, ctx_list=self.ctx_list, cast_to_int=self.cast_to_int)
        mev = mev_evaluator.evaluate(curr_rand_alphas, port_id=port_id, best=False, logfile=None)

        return mev

    def evaluate(self, sample, **kwargs):
        if self.use_repr:
            sample = self.translate_sample(sample)
            assert self.check_constraints(sample)
        
        transactions = reorder(self.transactions, sample)
        # print('\n'.join(map(str, transactions)))
        params = get_params(transactions)
        boundaries = []
        for p_name in params:
            boundaries.append(list(self.domain[p_name]))
        boundaries = np.asarray(boundaries)
        # print('=> current reordering sample:', transactions)

        mev_evaluator = MEV_evaluator(transactions, params, domain_scales=self.domain_scales, involved_dexes=self.involved_dexes, ctx_list=self.ctx_list, cast_to_int=self.cast_to_int)

        # perform adaptive sampling to optimize alpha values
        sampler = Gaussian_sampler(boundaries, minimum_num_good_samples=self.minimum_num_good_samples, random_loguniform=args.gauss_random_loguniform,
                                    u_random_portion=self.u_random_portion, local_portion=self.local_portion, cross_portion=self.cross_portion, pair_selection_method=self.pair_selection)

        #---------------- Run Sampling
        print('=> Starting alpha variable optimization')
        best_sample, best_mev = sampler.run_sampling(mev_evaluator.evaluate, num_samples=self.num_samples, n_iter=self.n_iter_gauss, minimize=False, 
                                            alpha_max=self.alpha_max, early_stopping=self.early_stopping, save_path=self.save_path, 
                                            n_parallel=self.n_parallel, plot_contour=False, executor=mp.Pool, param_names=params, verbose=False)
        print('=> optimal hyperparameters:', {p_name: v for p_name, v in zip(params, best_sample)})
        print('maximum MEV:', best_mev)
        print('-------------------------------------')
        
        subsample_info = {'n_subsamples':len(sampler.all_samples)}
        subsample_info.update({'variables': {p_name: v for p_name, v in zip(params, best_sample)}})
        return best_mev, subsample_info


def main(args, transaction, grid_search=False):
    if args.name is None:
        if args.reorder:
            if args.SA:
                args.name = f'SA_{args.n_iter}iter_{args.num_samples}nsamples'
            else:
                if args.p_swap_min != 0.0:
                    args.name = f'{args.n_iter}iter_{args.num_samples}nsamples_{args.u_random_portion}random_{args.parents_portion}parents_{args.p_swap_min}-{args.p_swap_max}p_swap'
                else:
                    args.name = f'{args.n_iter}iter_{args.num_samples}nsamples_{args.u_random_portion}random_{args.parents_portion}parents_{args.p_swap_max}p_swap'
                if args.swap_method == 'adjacent':
                    pass
                elif args.swap_method == 'adjacent_subset':
                    args.name += '_adjsubset'
                elif args.swap_method == 'adjacent_neighbor':
                    args.name += '_neighbor'
                else:
                    raise NotImplementedError
        elif grid_search:
            args.name = 'grid_search'
        else:
            args.name = f'{args.n_iter_gauss}iter_{args.num_samples_gauss}nsamples_{args.u_random_portion_gauss}random_{args.local_portion}local_{args.cross_portion}_cross'
    if 'amm' in transaction:
        problem_name = os.path.basename(os.path.abspath(os.path.join(transaction, os.pardir)))
        eth_pair = re.search('(0x[a-zA-Z0-9]+)', args.transactions).group(1)
    elif 'transactions_reordered' in transaction:
        problem_name = os.path.basename(os.path.abspath(os.path.join(transaction ,"../..")))
        eth_pair = re.search('(0x[a-zA-Z0-9]+)', args.transactions).group(1)
    else:
        problem_name = os.path.basename(transaction)
        eth_pair = None
    print(f'----------{eth_pair}_{problem_name}----------' if eth_pair is not None else f'----------{problem_name}----------')


    #---------------- Read input files and initialize the sampler
    transactions_f = open(transaction, 'r')
    transactions = transactions_f.readlines()

    # if len(transactions) >= 15:
    #     return

    domain_f = open(args.domain, 'r')
    domain = {}
    domain_scales = {}
    for line in domain_f.readlines():
        if line[0] == '#':
            continue
        tokens = line.strip().split(',')

        # TODO: add other currencies here
        lower_lim, upper_lim = float(tokens[1]), float(tokens[2])
        token_pair = args.domain.split('/')[-2]

        if len(tokens)==3:
            VALID_RANGE[token_pair] = min(1e6, upper_lim)
            if upper_lim > VALID_RANGE[token_pair]:
                domain_scales[tokens[0]] = upper_lim / VALID_RANGE[token_pair]
                upper_lim = VALID_RANGE[token_pair]
            else:
                domain_scales[tokens[0]] = 1.0
        else:
            assert len(tokens)==4
            domain_scales[tokens[0]] = float(tokens[3])

        domain[tokens[0]] = (lower_lim, upper_lim)
    print('domain:', domain)
    print('domain scales:', domain_scales)

    dir_name = 'artifacts_smooth{}_aave_oraclerelated'.format('_SA' if args.SA else '')
    if eth_pair is not None:
        dir_to_save = os.path.join(dir_name, eth_pair)
    else:
        dir_to_save = dir_name
    args.save_path = os.path.join(dir_to_save, problem_name, args.name)
    os.makedirs(args.save_path, exist_ok=True)  
    print('=> Saving artifacts to %s' % args.save_path)
    shutil.copyfile(transaction, os.path.join(args.save_path, 'transactions'))
    
    # dump git hash
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with open (os.path.join(dir_name, 'git_hash'), 'w') as f:
        f.write(sha)

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logger = logging.getLogger(__name__)

    cast_to_int = False
    involved_dexes = args.dexes
    for dex in involved_dexes:
        if 'uniswapv3' in dex:
            cast_to_int = True
            break
    if True: #try:      
        ctx = setup(transactions[0], capital=args.capital)
        #------------ generate contexts
        if args.n_parallel_gauss > 1:
            with mp.Pool() as pool:
                ctxes = list(pool.starmap(prepare_once, zip(repeat(ctx), repeat(transactions), range(args.n_parallel_gauss), repeat(involved_dexes))))
        else:
            ctxes = [prepare_once(ctx, transactions, 0, involved_dexes)]

        if not args.reorder:  
            if os.path.exists(os.path.join(args.save_path, 'history_info_0.pkl')):
                print('-------------------- already optimized')
                return

            params = get_params(transactions)
            logging.info(params)
            boundaries = []
            for p_name in params:
                boundaries.append(list(domain[p_name]))
            boundaries = np.asarray(boundaries)

            evaluator = MEV_evaluator(transactions, params, domain_scales=domain_scales, involved_dexes=involved_dexes, ctx_list=ctxes, cast_to_int=cast_to_int)

            if not grid_search:   # perform adaptive sampling to optimize alpha values
                log_file = f'final_results_{eth_pair}.txt' if eth_pair is not None else 'final_results.txt'

                sampler = Gaussian_sampler(boundaries, minimum_num_good_samples=int(0.5*args.num_samples_gauss), random_loguniform=args.gauss_random_loguniform,
                                            u_random_portion=args.u_random_portion_gauss, local_portion=args.local_portion, 
                                            cross_portion=args.cross_portion, pair_selection_method=args.pair_selection)

                #---------------- Run Sampling
                print('=> Starting optimization')
                best_sample, best_mev = sampler.run_sampling(evaluator.evaluate, num_samples=args.num_samples_gauss, n_iter=args.n_iter_gauss, minimize=False, 
                                                    alpha_max=args.alpha_max, early_stopping=args.early_stopping, save_path=args.save_path, 
                                                    n_parallel=args.n_parallel_gauss, plot_contour=args.plot_contour, executor=mp.Pool, param_names=params, verbose=True)
                if best_sample is None:
                    with open(os.path.join(dir_to_save, log_file), 'a') as f:
                        f.write('------------------- {} \n'.format(problem_name + '_random' if args.u_random_portion==1.0 else problem_name))
                        f.write(f'got None MEV, aborting optimization \n')
                    return

                evaluator_ = MEV_evaluator(transactions, params, domain_scales=domain_scales, involved_dexes=involved_dexes, ctx_list=ctxes, cast_to_int=cast_to_int)
                mev = evaluator_.evaluate(best_sample, port_id=0, best=True, logfile=os.path.join(args.save_path, 'transactions_optimized'))
                print(f'expected {best_mev}, got {mev}')
                assert mev == best_mev
                
                print('=> optimal hyperparameters:', {p_name: v for p_name, v in zip(params, best_sample)})
                print('maximum MEV:', best_mev)
                with open(os.path.join(dir_to_save, log_file), 'a') as f:
                    f.write('------------------- {} \n'.format(problem_name + '_random' if args.u_random_portion_gauss==1.0 else problem_name))
                    f.write(f'max MEV: {best_mev} \n')
                    f.write('params: {} \n'.format({p_name: v for p_name, v in zip(params, best_sample)})) 
            
            else:  # perform exhaustive grid search to optimize alpha values
                path_to_save = args.save_path
                os.makedirs(path_to_save, exist_ok=True)

                if not os.path.exists(os.path.join(path_to_save, 'scores.pkl')):
                    grid = {}
                    total = 1
                    for p_name in params:
                        if p_name in ['alpha1', 'alpha2']:
                            count = 20
                        else:
                            count = 5
                        grid[p_name] = np.linspace(domain[p_name][0], domain[p_name][-1], num=count)
                        total *= len(grid[p_name])
                    samples = np.vstack(np.meshgrid(*[grid[p_name] for p_name in params])).reshape(len(params), -1).T
                    with open(os.path.join(path_to_save, 'samples.pkl'), 'wb') as f:
                        pickle.dump([params, samples], f)
                    
                    n_parallel = 4
                    n_batches = len(samples)//n_parallel if len(samples)%n_parallel==0 else (len(samples)//n_parallel)+1
                    scores = np.zeros(len(samples))
                    
                    with tqdm(total=n_batches) as pbar:
                        for i in range(n_batches):
                            batch_samples = samples[i*n_parallel:(i+1)*n_parallel]

                            with mp.Pool() as e:
                                scores[i*n_parallel:(i+1)*n_parallel] = list(e.map(evaluator.evaluate, batch_samples))
                            
                            pbar.update(1)
                            pbar.set_description('batch %s/%s (samples %s..%s/%s)'%(i+1, len(samples)//n_parallel, i*n_parallel, \
                                                            (i+1)*n_parallel, len(samples))) 

                            if i % 200==0:
                                print('=> saving history so far')
                                with open(os.path.join(path_to_save, 'scores.pkl'), 'wb') as f:
                                    pickle.dump(scores, f)
                else:
                    with open(os.path.join(path_to_save, 'scores.pkl'), 'rb') as f:
                        scores = pickle.load(f)
                    with open(os.path.join(path_to_save, 'samples.pkl'), 'rb') as f:
                        log = pickle.load(f)
                    params, samples = log[0], log[1]

                    idx = np.argmax(scores)
                    print('=> optimal hyperparameters:', {p_name: v for p_name, v in zip(params, samples[idx])})
                    print('maximum MEV:', scores[idx])

        else:
            log_file = f'final_results_reorder_{eth_pair}.txt' if eth_pair is not None else 'final_results_reorder.txt'
            if args.SA:
                text = problem_name + 'SA'
            else:
                text = problem_name + '_random' if args.u_random_portion==1.0 else problem_name
            if os.path.exists(os.path.join(args.save_path, 'transactions_optimized')) or '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48/15001234' in transaction \
                                                                                    or '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48/15424706' in transaction:
                with open(os.path.join(dir_to_save, log_file), 'a') as f:
                    f.write(f'------------------- {text} \n')
                    f.write(f'{eth_pair}_{problem_name} already optimized \n')
                print(f'{eth_pair}_{problem_name} already optimized \n')
                return
            else:
                shutil.rmtree(args.save_path)
                os.makedirs(args.save_path)
        
            gt_order = get_groundtruth_order(transactions[1:], include_miner=True)

            n_reorders = math.factorial(len(transactions)-1)
            if n_reorders < args.num_samples:
                args.num_samples = n_reorders

            evaluator = Reorder_evaluator(transactions, domain, domain_scales, args.n_iter_gauss, args.num_samples_gauss, int(0.5*args.num_samples_gauss), 
                                            args.u_random_portion_gauss, args.local_portion, args.cross_portion, args.pair_selection, 
                                            args.alpha_max, args.early_stopping, args.save_path, n_parallel=args.n_parallel_gauss, involved_dexes=involved_dexes, ctx_list=ctxes,
                                            use_repr=True, groundtruth_order=gt_order, cast_to_int=cast_to_int)
            

            if args.SA:
                sampler = SA_sampler(evaluator.evaluate, length=len(transactions)-1, groundtruth_order=gt_order)

                #---------------- Find out which transactions' location matters the most
                get_MEV_effects = True
                orig_order = sampler.order_repr
                MEV_diffs = {}
                if get_MEV_effects:
                    params = get_params(transactions)
                    boundaries = []
                    for p_name in params:
                        boundaries.append(list(domain[p_name]))
                    boundaries = np.asarray(boundaries)
                    rand_alphas = {k: np.random.uniform(boundaries[i][0], boundaries[i][1]) for i,k in enumerate(params)}

                    max_diff = -np.Infinity
                    for orig_idx, t in enumerate(orig_order):
                        min_MEV = np.Infinity
                        max_MEV = -np.Infinity
                        new_idx = 0
                        while True:
                            batch_samples = []
                            for _ in range(args.n_parallel_gauss):
                                if new_idx >= len(orig_order):
                                    break
                                res_transactions = copy.deepcopy(orig_order)
                                res_transactions.pop(orig_idx)
                                curr_order = []
                                for i in range(len(orig_order)):
                                    if i==new_idx:
                                        curr_order.append(t)
                                    else:
                                        curr_order.append(res_transactions.pop(0))
                                batch_samples.append(curr_order)
                                new_idx += 1
                            with mp.Pool() as e:
                                batch_output = list(e.starmap(evaluator.evaluate_randomAlphas, zip(batch_samples, [rand_alphas]*len(batch_samples),
                                                                                                    range(args.n_parallel_gauss))))
                            for curr_mev in batch_output:
                                if curr_mev is None:
                                    continue
                                if curr_mev < min_MEV:
                                    min_MEV = curr_mev
                                if curr_mev > max_MEV:
                                    max_MEV = curr_mev
                            if new_idx >= len(orig_order):
                                break
                        
                        # for new_idx in range(len(orig_order)):
                        #     res_transactions = copy.deepcopy(orig_order)
                        #     res_transactions.pop(orig_idx)
                        #     curr_order = []
                        #     for i in range(len(orig_order)):
                        #         if i==new_idx:
                        #             curr_order.append(t)
                        #         else:
                        #             curr_order.append(res_transactions.pop(0))
                        #     curr_mev = evaluator.evaluate_randomAlphas(curr_order)
                        #     if curr_mev < min_MEV:
                        #         min_MEV = curr_mev
                        #     if curr_mev > max_MEV:
                        #         max_MEV = curr_mev
                        
                        if t in MEV_diffs:
                            MEV_diffs[t].append(max_MEV - min_MEV + 1e-3)
                        else:
                            MEV_diffs[t] = [max_MEV - min_MEV + 1e-3]
                        if (max_MEV - min_MEV) > max_diff:
                            max_diff = max_MEV - min_MEV
                        print(f'maximum mev difference for tx {orig_idx} was {MEV_diffs[t][-1]}')
                    for k, v in MEV_diffs.items():
                        if k in ['M0', 'M1']:
                            MEV_diffs[k] = [max_diff] * len(v)
                else:
                    for orig_idx, t in enumerate(orig_order):
                        if t in MEV_diffs:
                            MEV_diffs[t].append(1.)
                        else:
                            MEV_diffs[t] = [1.]
                print('MEV_diffs:', MEV_diffs)
                sampler.set_MEV_effects(MEV_diffs)

                #---------------- Run Sampling
                print('=> Starting reordering optimization')
                best_order, best_mev, subsample_info = sampler.run_sampling(num_samples=args.num_samples, n_iter=args.n_iter, early_stopping=10, 
                                                                            save_path=args.save_path, param_names=None, verbose=True)
            else:
                sampler = RandomOrder_sampler(length=len(transactions)-1, minimum_num_good_samples=int(0.5*args.num_samples), 
                                        p_swap_min=args.p_swap_min, p_swap_max=args.p_swap_max, 
                                        u_random_portion=args.u_random_portion, parents_portion=args.parents_portion,
                                        swap_method=args.swap_method, groundtruth_order=gt_order)
                #---------------- Run Sampling
                print('=> Starting reordering optimization')
                best_order, best_mev, subsample_info = sampler.run_sampling(evaluator.evaluate, num_samples=args.num_samples, n_iter=args.n_iter, minimize=False, 
                                                    alpha_max=args.alpha_max, early_stopping=5, save_path=args.save_path, 
                                                    n_parallel=args.n_parallel, plot_contour=False, executor=mp.Pool, param_names=None, verbose=True)
            if best_order is None:
                with open(os.path.join(dir_to_save, log_file), 'a') as f:
                    f.write(f'------------------- {text} \n')
                    f.write(f'got None MEV, aborting optimization \n')
                print(f'got None MEV, aborting optimization \n')
                return
                    
            # check that the variable values are correct
            best_variables = subsample_info['variables']
            vars = list(best_variables.keys())
            if evaluator.use_repr:
                best_order = evaluator.translate_sample(best_order)
                assert evaluator.check_constraints(best_order)
            evaluator_ = MEV_evaluator(reorder(transactions, best_order), vars, domain_scales, involved_dexes=involved_dexes, ctx_list=ctxes, cast_to_int=cast_to_int)
            mev = evaluator_.evaluate([best_variables[k] for k in vars], port_id=0, best=True, logfile=os.path.join(args.save_path, 'transactions_optimized'))
            print(f'expected {best_mev}, got {mev}')
            assert mev == best_mev
            
            print('=> optimal transaction order:', reorder(transactions, best_order))
            print('=> optimal variables:', best_variables)
            print('maximum MEV:', best_mev)
            with open(os.path.join(dir_to_save, log_file), 'a') as f:
                f.write(f'------------------- {text} \n')
                f.write(f'max MEV: {best_mev} \n')
                f.write('=> optimal transaction order: {} \n'.format(reorder(transactions, best_order)))
                f.write(f'params: {best_variables} \n') 
    
    # except:
    #     if args.reorder:
    #         log_file = f'final_results_reorder_{eth_pair}.txt' if eth_pair is not None else 'final_results_reorder.txt'   
    #     else:
    #         log_file = f'final_results_{eth_pair}.txt' if eth_pair is not None else 'final_results.txt'
    #     with open(os.path.join(dir_to_save, log_file), 'a') as f:
    #         f.write('------------------- error occured when running {} \n'.format(problem_name + '_random' if args.u_random_portion_gauss==1.0 else problem_name))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Optimization')

    #------------ Arguments for transactios
    parser.add_argument('-v', '--verbose', help="Be verbose", action="store_const", dest="loglevel", const=logging.INFO, default=logging.WARNING)
    parser.add_argument('-t', '--transactions', help="Input File path containing parametric transactions")
    parser.add_argument('-d', '--domain', help="Input File path containing domains for parameters")
    parser.add_argument('--ignore', help="Input File path containing problem names to ignore", default=None)
    parser.add_argument('--grid', action='store_true', help='do grid search instead of sampling')
    parser.add_argument('--dexes', nargs='+', help='space separated list of dexes involved in optimization', required=True)
    parser.add_argument('--capital', default=1000, type=int, help='miner captial', required=True)
    
    #------------ Arguments for transaction reordering
    parser.add_argument('--reorder', action='store_true', help='optimize reordering of transactions')
    parser.add_argument('--SA', action='store_true', help='use simulated annealing to optimize reordering')
    parser.add_argument('--n_iter', default=50, type=int, help='number of optimization iterations (default: 50)')
    parser.add_argument('--num_samples', default=50, type=int, help='per-iteration sample size for reordering (default: 50)')
    parser.add_argument('--u_random_portion', default=0.2, type=float, help='portion of ranodm reorderings (default:0.2)')
    parser.add_argument('--parents_portion', default=0.1, type=float, help='portion of good samples to keep for the next round (default:0.1)')
    parser.add_argument('--p_swap_min', default=0.0, type=float, help='minimum probability of per-element swap (default:0.0)')
    parser.add_argument('--p_swap_max', default=0.5, type=float, help='maximum probability of per-element swap (default:0.5)')
    parser.add_argument('--swap_method', default='adjacent_neighbor', type=str, help='choose swapping method from [adjacent, adjacent_neighbor, adjacent_subset] (default:adjacent)')

    #------------ Arguments for adaptive sampling
    parser.add_argument('--name', default=None, help='name of the experiment (default: None)')
    parser.add_argument('--plot_contour', action='store_true', help='if selected, the sampler will save contours of the objective function along with per-iteration samples')
    parser.add_argument('--seed', default=0, type=int, help='random seed (default: 0)')
	
    #------ Sampler parameters
    parser.add_argument('--num_samples_gauss', default=50, type=int, help='per-iteration gaussian sample size (default: 50)')
    parser.add_argument('--dim', type=int, help='dimensionality of the search-space (default: None)')
    parser.add_argument('--n_iter_gauss', default=50, type=int, help='number of sampling iterations for finding alphas(default: 50)')
    parser.add_argument('--n_parallel', default=1, type=int, help='number of cores for parallel evaluations (default:1)')
    parser.add_argument('--alpha_max', default=1.0, type=float, help='alpha_max parameter (default:1.0)')
    parser.add_argument('--early_stopping', default=10, type=int, help='number of iterations without improvement to activate early stopping (default: 10)')

    #------ Gaussian Sampler parameters
    parser.add_argument('--gauss_random_loguniform', action='store_true', help='whether to use loguniform or uniform sampler for random samples.')
    parser.add_argument('--u_random_portion_gauss', default=0.2, type=float, help='portion of samples to take unifromly random from the entire space (default:0.2)')
    parser.add_argument('--local_portion', default=0.4, type=float, help='portion of samples to take from gaussians using the Local method (default:0.4)')
    parser.add_argument('--cross_portion', default=0.4, type=float, help='portion of samples to take from gaussians using the Cross method (default:0.4)')
    parser.add_argument('--pair_selection', default='top_and_random', type=str, help='how to select sample pairs for crossing, choose from [random,top_scores,top_and_nearest,top_and_furthest,top_and_random] (default:top_and_random)')
    parser.add_argument('--n_parallel_gauss', default=44, type=int, help='number of cores for parallel evaluations of Gaussian sampler (default:44)')

    args = parser.parse_args()  
    # np.random.seed(args.seed)
    print(args.dexes)

    invalid_problems = []
    if args.ignore is not None:
        with open(args.ignore, 'r') as f:
            invalid_problems = f.readlines()

    file_pattern = '_reduced'
    if os.path.isdir(args.transactions):
        all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(args.transactions) for f in filenames if file_pattern in f] 
                        # if file_pattern in f and int(os.path.basename(dp))>=15.e6] 
                        # if file_pattern in f and int(os.path.basename(dp))>=14.e6] 
                        # if file_pattern in f and int(os.path.basename(dp))>=13e6 and int(os.path.basename(dp))<14.e6]
        all_files = np.sort(all_files)
        print(f'found {len(all_files)} files for optimization')

        # with open('artifacts_smooth_sushiswap/info_summary.yaml', 'r') as f:
        #     sushiswap_results = yaml.safe_load(f)

        for transaction in all_files:
            if f'{transaction}\n' in invalid_problems:
                print(f'-------------- ignoring {transaction}')
                continue

            # p_key = '{}/{}'.format(transaction.split('/')[-3], transaction.split('/')[-2])
            # if not p_key in sushiswap_results.keys():
            #     print(f'-------------- ignoring {p_key}')
            #     continue

            main(args, transaction, grid_search=args.grid)
    else:
        if not f'{args.transactions}\n' in invalid_problems:
            assert os.path.isfile(args.transactions)
            main(args, args.transactions, grid_search=args.grid)




    

