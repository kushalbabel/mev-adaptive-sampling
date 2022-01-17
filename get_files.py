import os
import numpy as np

def gather_results(path):
    def gather_result_paths(path):
        paths = []
        if os.path.isfile(path):
                return [path]
        else:
            for d in os.listdir(path):
                paths += gather_result_paths(os.path.join(path, d))
        return paths

    paths = gather_result_paths(path)
    lengths = []
    for p in paths:
        transactions_f = open(p, 'r')
        transactions = transactions_f.readlines()
        lengths.append(len(transactions)-1)
    
    return lengths

path_to_tests = '/home/kb742/mev-adaptive-sampling/'
testset = 'tests'
lengths = gather_results(os.path.join(path_to_tests, testset))
print(np.min(lengths), np.max(lengths))
print(np.median(lengths))
print(np.math.factorial(int(np.max(lengths)))/1000)