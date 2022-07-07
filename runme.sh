# TRANSACTIONS='/home/kb742/mev-adaptive-sampling/tests'
# DOMAIN='/home/kb742/mev-adaptive-sampling/domain'
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --u_random_portion_gauss 1.0 --local_portion 0.0 --cross_portion 0.0 --early_stopping 1000
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --early_stopping 1000
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 20 --num_samples 10 --parents_portion 0. --early_stopping 1000
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 30 --num_samples 20 --parents_portion 0.0 --p_swap_max 0.8 --p_swap_min 0.1
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 30 --num_samples 20 --u_random_portion 1. --parents_portion 0. --early_stopping 1000

# TRANSACTIONS='/home/kb742/mev-adaptive-sampling/eth_token_tests/0x397ff1542f962076d0bfe58ea045ffa2d347aca0/13076406'  # example problem which finds a new technique
TRANSACTIONS='/home/kb742/mev-adaptive-sampling/eth_token_tests/0x795065dcc9f64b5614c407a6efdc400da6221fb0'
DOMAIN='/home/kb742/mev-adaptive-sampling/eth_token_tests/0x795065dcc9f64b5614c407a6efdc400da6221fb0/domain'
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --u_random_portion_gauss 1.0 --local_portion 0.0 --cross_portion 0.0 --num_samples_gauss 120
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --u_random_portion_gauss 0.5 --local_portion 0.25 --cross_portion 0.25 --early_stopping 1000
python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 5 --num_samples 10 --parents_portion 0.0 --p_swap_max 0.8 --p_swap_min 0.1 --num_samples_gauss 44
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 5 --num_samples 10 --u_random_portion 1. --parents_portion 0.0 --p_swap_max 0.8 --p_swap_min 0.1 --num_samples_gauss 44 --early_stopping 1000