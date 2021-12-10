import os
import argparse
import logging
import numpy as np
import multiprocessing as mp

from simulate import simulate
from sampling_utils import Gaussian_sampler


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


def main(args, transaction):
    if args.name is None:
        args.name = f'iter{args.n_iter}_{args.num_samples}nsamples_{args.u_random_portion}random_{args.local_portion}local_{args.cross_portion}_cross'
    problem_name = os.path.basename(transaction)
    print(f'----------{problem_name}----------')

    args.save_path = os.path.join('artifacts', problem_name, args.name)
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

    params = get_params(transactions)
    logging.info(params)
    boundaries = []
    for p_name in params:
        boundaries.append(list(domain[p_name]))
    boundaries = np.asarray(boundaries)

    sampler = Gaussian_sampler(boundaries, minimum_num_good_samples=int(0.5*args.num_samples), 
                                u_random_portion=args.u_random_portion, local_portion=args.local_portion, cross_portion=args.cross_portion, pair_selection_method=args.pair_selection)
    evaluator = MEV_evaluator(transactions, params)

    #---------------- Run Sampling
    print('=> Starting optimization')
    best_sample, best_mev = sampler.run_sampling(evaluator.evaluate, args.num_samples, args.n_iter, args.minimize, args.alpha_max, early_stopping=args.early_stopping, 
                                        save_path=args.save_path, n_parallel=args.n_parallel, plot_contour=args.plot_contour, 
                                        executor=mp.Pool, param_names=params)
    print('=> optimal hyperparameters:', {p_name: v for p_name, v in zip(params, best_sample)})
    print('maximum MEV:', best_mev)

    with open('final_results.txt', 'a') as f:
        f.write(f'------------------- {problem_name} \n')
        f.write(f'max MEV: {best_mev} \n')
        f.write('params: {} \n'.format({p_name: v for p_name, v in zip(params, best_sample)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Optimization')

    #------------ Arguments for transactios
    parser.add_argument('-v', '--verbose', help="Be verbose", action="store_const", dest="loglevel", const=logging.INFO, default=logging.WARNING)
    parser.add_argument('-t', '--transactions', help="Input File path containing parametric transactions", required=True)
    parser.add_argument('-d', '--domain', help="Input File path containing domains for parameters", required=True)

    #------------ Arguments for adaptive sampling
    parser.add_argument('--name', default=None, help='name of the experiment (default: None)')
    parser.add_argument('--minimize', action='store_true', help='if selected, the function will be minimized, otherwise maximized')
    parser.add_argument('--test_fn', type=str, help='(optional) choose from common optimization test functions [rastrigin, ]')
    parser.add_argument('--plot_contour', action='store_true', help='if selected, the sampler will save contours of the objective function along with per-iteration samples')
    parser.add_argument('--seed', default=0, type=int, help='random seed (default: 0)')
	
    #------ Sampler parameters
    parser.add_argument('--num_samples', default=50, type=int, help='per-iteration sample size (default: 50)')
    parser.add_argument('--dim', type=int, help='dimensionality of the search-space (default: None)')
    parser.add_argument('--n_iter', default=50, type=int, help='number of optimization iterations (default: 50)')
    parser.add_argument('--n_parallel', default=1, type=int, help='number of cores for parallel evaluations (default:1)')
    parser.add_argument('--alpha_max', default=1.0, type=float, help='alpha_max parameter (default:1.0)')
    parser.add_argument('--early_stopping', default=10, type=int, help='number of iterations without improvement to activate early stopping (default: 1000)')

    #------ Gaussian Sampler parameters
    parser.add_argument('--u_random_portion', default=0.2, type=float, help='portion of samples to take unifromly random from the entire space (default:0.2)')
    parser.add_argument('--local_portion', default=0.4, type=float, help='portion of samples to take from gaussians using the Local method (default:0.4)')
    parser.add_argument('--cross_portion', default=0.4, type=float, help='portion of samples to take from gaussians using the Cross method (default:0.4)')
    parser.add_argument('--pair_selection', default='top_and_random', type=str, help='how to select sample pairs for crossing, choose from [random,top_scores,top_and_nearest,top_and_furthest,top_and_random] (default:top_and_random)')

    args = parser.parse_args()  
    # np.random.seed(args.seed)
    
    if os.path.isdir(args.transactions):
        all_files = [os.path.join(args.transactions, f) for f in os.listdir(args.transactions) if os.path.isfile(os.path.join(args.transactions, f))]
        print(f'found {len(all_files)} files for optimization')

        for transaction in all_files:
            main(args, transaction)

    else:
        assert os.path.isfile(args.transactions)
        main(args, args.transactions)




    

