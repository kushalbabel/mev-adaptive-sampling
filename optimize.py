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

def get_params(transactions):
    params = set()
    for transaction in transactions:
        vals = transaction.split(',')
        for val in vals:
            if 'alpha' in val:
                params.add(val)
    return list(params)

def next_sample(params, domain):
    sample = {}
    for param in params:
        sample[param] = domain[param][0] # naively try the start of range
    return sample

def substitute(transactions, sample):
    datum = []
    for transaction in transactions:
        transaction = transaction.strip()
        for param in sample:
            transaction = transaction.replace(param, str(sample[param]))
        datum.append(transaction)
    return datum

# transactions: parametric list of transactions (a transaction is csv values)
class MEV_evaluator(object):
    def __init__(self, transactions, params):
        self.transactions = transactions
        self.params = params
        
    def evaluate(self, sample):
        # sample is a vector that has the values for parameter names in self.params    
        sample_dict = {p_name: v for p_name, v in zip(self.params, sample)}
        datum = substitute(self.transactions, sample_dict)
        logging.info(datum)
        mev = simulate(datum)

        return mev


def reorder(transactions, order):
    '''
        function to reorder a set of transactions, except for the first one
    '''
    order = order.astype(np.int32)
    reordered_transactions = [transactions[0]] + [transactions[i+1] for i in order]
    return reordered_transactions


class Reorder_evaluator(object):
    def __init__(self, transactions, domain, n_iter_gauss, num_samples, minimum_num_good_samples, u_random_portion, local_portion, cross_portion, 
                pair_selection, alpha_max, early_stopping, save_path, n_parallel):
        self.transactions = transactions
        self.domain = domain

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

    def evaluate(self, sample):
        transactions = reorder(self.transactions, sample)

        params = get_params(transactions)
        boundaries = []
        for p_name in params:
            boundaries.append(list(self.domain[p_name]))
        boundaries = np.asarray(boundaries)

        mev_evaluator = MEV_evaluator(transactions, params)

        # perform adaptive sampling to optimize alpha values
        sampler = Gaussian_sampler(boundaries, minimum_num_good_samples=self.minimum_num_good_samples, 
                                    u_random_portion=self.u_random_portion, local_portion=self.local_portion, cross_portion=self.cross_portion, pair_selection_method=self.pair_selection)

        #---------------- Run Sampling
        print('=> Starting alpha variable optimization')
        best_sample, best_mev = sampler.run_sampling(mev_evaluator.evaluate, num_samples=self.num_samples, n_iter=self.n_iter_gauss, minimize=False, 
                                            alpha_max=self.alpha_max, early_stopping=self.early_stopping, save_path=self.save_path, 
                                            n_parallel=self.n_parallel, plot_contour=False, executor=mp.Pool, param_names=params)
        print('=> optimal hyperparameters:', {p_name: v for p_name, v in zip(params, best_sample)})
        print('maximum MEV:', best_mev)
        print('-------------------------------------')
        
        return best_mev, {p_name: v for p_name, v in zip(params, best_sample)}


def main(args, transaction, grid_search=False):
    if args.name is None:
        if args.reorder:
            if args.p_swap_min != 0.0:
                args.name = f'{args.n_iter}iter_{args.num_samples}nsamples_{args.u_random_portion}random_{args.parents_portion}parents_{args.p_swap_min}-{args.p_swap_max}p_swap'
            else:
                args.name = f'{args.n_iter}iter_{args.num_samples}nsamples_{args.u_random_portion}random_{args.parents_portion}parents_{args.p_swap_max}p_swap'
            if args.swap_method == 'adjacent':
                pass
            if args.swap_method == 'adjacent_subset':
                args.name += '_adjsubset'
            else:
                raise NotImplementedError
        elif grid_search:
            args.name = 'grid_search'
        else:
            args.name = f'{args.n_iter_gauss}iter_{args.num_samples_gauss}nsamples_{args.u_random_portion_gauss}random_{args.local_portion}local_{args.cross_portion}_cross'
    problem_name = os.path.basename(transaction)
    testset = os.path.basename(os.path.dirname(transaction))
    print(f'----------{problem_name}----------')

    args.save_path = os.path.join('artifacts2', testset, problem_name, args.name)
    print('=> Saving artifacts to %s' % args.save_path)

    os.makedirs(args.save_path, exist_ok=True)  
    logging.basicConfig(level=args.loglevel, format='%(message)s')

    logger = logging.getLogger(__name__)

    #---------------- Read input files and initialize the sampler
    transactions_f = open(transaction, 'r')
    transactions = transactions_f.readlines()

    domain_f = open(args.domain, 'r')
    domain = {}
    for line in domain_f.readlines():
        tokens = line.strip().split(',')
        domain[tokens[0]] = (float(tokens[1]), float(tokens[2]))
    print(domain)

    if not args.reorder:  
        params = get_params(transactions)
        logging.info(params)
        boundaries = []
        for p_name in params:
            boundaries.append(list(domain[p_name]))
        boundaries = np.asarray(boundaries)

        evaluator = MEV_evaluator(transactions, params)

        if not grid_search:   # perform adaptive sampling to optimize alpha values
            sampler = Gaussian_sampler(boundaries, minimum_num_good_samples=int(0.5*args.num_samples_gauss), 
                                        u_random_portion=args.u_random_portion_gauss, local_portion=args.local_portion, cross_portion=args.cross_portion, pair_selection_method=args.pair_selection)

            #---------------- Run Sampling
            print('=> Starting optimization')
            best_sample, best_mev = sampler.run_sampling(evaluator.evaluate, num_samples=args.num_samples_gauss, n_iter=args.n_iter_gauss, minimize=False, 
                                                alpha_max=args.alpha_max, early_stopping=args.early_stopping, save_path=args.save_path, 
                                                n_parallel=args.n_parallel, plot_contour=args.plot_contour, executor=mp.Pool, param_names=params)
            print('=> optimal hyperparameters:', {p_name: v for p_name, v in zip(params, best_sample)})
            print('maximum MEV:', best_mev)

            with open('final_results.txt', 'a') as f:
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

                print(len(scores))
                idx = np.argmax(scores)
                print(np.sum(scores < 0))
                print('=> optimal hyperparameters:', {p_name: v for p_name, v in zip(params, samples[idx])})
                print('maximum MEV:', scores[idx])

    else:
        sampler = RandomOrder_sampler(length=len(transactions)-1, minimum_num_good_samples=int(0.5*args.num_samples), 
                                    p_swap_min=args.p_swap_min, p_swap_max=args.p_swap_max, 
                                    u_random_portion=args.u_random_portion, parents_portion=args.parents_portion,
                                    swap_method=args.swap_method)

        evaluator = Reorder_evaluator(transactions, domain, args.n_iter_gauss, args.num_samples_gauss, int(0.5*args.num_samples_gauss), 
                                        args.u_random_portion_gauss, args.local_portion, args.cross_portion, args.pair_selection, 
                                        args.alpha_max, args.early_stopping, args.save_path, args.n_parallel)
        #---------------- Run Sampling
        print('=> Starting reordering optimization')
        best_order, best_mev, best_variables = sampler.run_sampling(evaluator.evaluate, num_samples=args.num_samples, n_iter=args.n_iter, minimize=False, 
                                            alpha_max=args.alpha_max, early_stopping=1000, save_path=args.save_path, 
                                            n_parallel=args.n_parallel, plot_contour=False, executor=mp.Pool, param_names=None, verbose=True)
        
        # check that the variable values are correct
        vars = list(best_variables.keys())
        evaluator = MEV_evaluator(reorder(transactions, best_order), vars)
        mev = evaluator.evaluate([best_variables[k] for k in vars])
        print(f'expected {best_mev}, got {mev}')
        assert mev == best_mev
        
        print('=> optimal transaction order:', reorder(transactions, best_order))
        print('=> optimal variables:', best_variables)
        print('maximum MEV:', best_mev)

        with open('final_results_reorder.txt', 'a') as f:
            f.write('------------------- {} \n'.format(problem_name + '_random' if args.u_random_portion==1.0 else problem_name))
            f.write(f'max MEV: {best_mev} \n')
            f.write('=> optimal transaction order: {} \n'.format(reorder(transactions, best_order)))
            f.write(f'params: {best_variables} \n') 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Optimization')

    #------------ Arguments for transactios
    parser.add_argument('-v', '--verbose', help="Be verbose", action="store_const", dest="loglevel", const=logging.INFO, default=logging.WARNING)
    parser.add_argument('-t', '--transactions', help="Input File path containing parametric transactions")
    parser.add_argument('-d', '--domain', help="Input File path containing domains for parameters")
    parser.add_argument('--grid', action='store_true', help='do grid search instead of sampling')
    
    #------------ Arguments for transaction reordering
    parser.add_argument('--reorder', action='store_true', help='optimize reordering of transactions')
    parser.add_argument('--n_iter', default=50, type=int, help='number of optimization iterations (default: 50)')
    parser.add_argument('--num_samples', default=50, type=int, help='per-iteration sample size for reordering (default: 50)')
    parser.add_argument('--u_random_portion', default=0.2, type=float, help='portion of ranodm reorderings (default:0.2)')
    parser.add_argument('--parents_portion', default=0.1, type=float, help='portion of good samples to keep for the next round (default:0.1)')
    parser.add_argument('--p_swap_min', default=0.0, type=float, help='minimum probability of per-element swap (default:0.0)')
    parser.add_argument('--p_swap_max', default=0.5, type=float, help='maximum probability of per-element swap (default:0.5)')
    parser.add_argument('swap_method', default='adjacent', type=str, help='choose swapping method from [adjacent, adjacent_subset] (default:adjacent)')

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
    parser.add_argument('--u_random_portion_gauss', default=0.2, type=float, help='portion of samples to take unifromly random from the entire space (default:0.2)')
    parser.add_argument('--local_portion', default=0.4, type=float, help='portion of samples to take from gaussians using the Local method (default:0.4)')
    parser.add_argument('--cross_portion', default=0.4, type=float, help='portion of samples to take from gaussians using the Cross method (default:0.4)')
    parser.add_argument('--pair_selection', default='top_and_random', type=str, help='how to select sample pairs for crossing, choose from [random,top_scores,top_and_nearest,top_and_furthest,top_and_random] (default:top_and_random)')

    args = parser.parse_args()  
    # np.random.seed(args.seed)

    ntransactions = 30
    if os.path.isdir(args.transactions):
        all_files = [os.path.join(args.transactions, f) for f in os.listdir(args.transactions) if os.path.isfile(os.path.join(args.transactions, f))]
        all_files = np.sort(all_files)[2*ntransactions:]
        print(f'found {len(all_files)} files for optimization')
        
        for transaction in all_files:
            try:
                main(args, transaction, grid_search=args.grid)

            except:
                print(f'======== error occured when running {transaction}')
                continue

    else:
        assert os.path.isfile(args.transactions)
        main(args, args.transactions, grid_search=args.grid)




    

