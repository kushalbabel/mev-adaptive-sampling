import argparse
import logging
from simulate import simulate

def get_params(transactions):
    params = set()
    for transaction in transactions:
        vals = transaction.split(',')
        for val in vals:
            if 'alpha' in val:
                params.add(val)
    return params

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
            transaction = transaction.replace(param, sample[param])
        datum.append(transaction)
    return datum

# transactions: parametric list of transactions (a transaction is csv values)
def adaptive_sampling(transactions  , domain):
    logging.info(domain)
    params = get_params(transactions)
    logging.info(params)
    for i in range(1,10):
        sample = next_sample(params, domain)
        datum = substitute(transactions, sample)
        logging.info(datum)
        mev = simulate(datum)
    return mev


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Optimization')

    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
        default=logging.WARNING
    )

    parser.add_argument(
        '-t', '--transactions',
        help="Input File path containing parametric transactions",
        required=True
    )

    parser.add_argument(
        '-d', '--domain',
        help="Input File path containing domains for parameters",
        required=True
    )


    args = parser.parse_args()    
    logging.basicConfig(level=args.loglevel, format='%(message)s')

    logger = logging.getLogger(__name__)

    transactions_f = open(args.transactions, 'r')
    domain_f = open(args.domain, 'r')
    domain = {}
    for line in domain_f.readlines():
        tokens = line.strip().split(',')
        domain[tokens[0]] = (tokens[1], tokens[2])
    mev = adaptive_sampling(transactions_f.readlines(), domain)
    print(mev)