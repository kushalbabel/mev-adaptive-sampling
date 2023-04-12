# TRANSACTIONS='/home/kb742/mev-adaptive-sampling/tests'
# DOMAIN='/home/kb742/mev-adaptive-sampling/domain'
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --u_random_portion_gauss 1.0 --local_portion 0.0 --cross_portion 0.0 --early_stopping 1000
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --early_stopping 1000
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 20 --num_samples 10 --parents_portion 0. --early_stopping 1000
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 30 --num_samples 20 --parents_portion 0.0 --p_swap_max 0.8 --p_swap_min 0.1
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 30 --num_samples 20 --u_random_portion 1. --parents_portion 0. --early_stopping 1000

# TRANSACTIONS='/home/kb742/mev-adaptive-sampling/eth_token_tests/0x397ff1542f962076d0bfe58ea045ffa2d347aca0/13076406'  # example problem which finds a new technique
# TRANSACTIONS='/home/kb742/mev-adaptive-sampling/eth_token_tests/0x795065dcc9f64b5614c407a6efdc400da6221fb0/13179357'
# TRANSACTIONS='/home/kb742/mev-adaptive-sampling/eth_token_tests_v2composition/0xceff51756c56ceffca006cd410b03ffc46dd3a58/13332889'
# DOMAIN='/home/kb742/mev-adaptive-sampling/eth_token_tests_v2composition/0xceff51756c56ceffca006cd410b03ffc46dd3a58/domain'
TRANSACTIONS='/home/kb742/mev-adaptive-sampling/eth_token_tests_uniswapv3/0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48/15006831'
DOMAIN='/home/kb742/mev-adaptive-sampling/eth_token_tests_uniswapv3/0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48/domain'
DEXES='uniswapv3'
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --u_random_portion_gauss 1.0 --local_portion 0.0 --cross_portion 0.0 --num_samples_gauss 120
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --num_samples_gauss 44
python optimize.py -t $TRANSACTIONS -d $DOMAIN --dexes $DEXES --reorder --n_iter 5 --num_samples 10 --parents_portion 0.0 --p_swap_max 0.8 --p_swap_min 0.1 --num_samples_gauss 40 --n_parallel_gauss 40
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 5 --num_samples 10 --u_random_portion 1. --parents_portion 0.0 --p_swap_max 0.8 --p_swap_min 0.1 --num_samples_gauss 44 --early_stopping 1000
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --SA --n_iter 50 --num_samples 1