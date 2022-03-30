# TRANSACTIONS='/home/kb742/mev-adaptive-sampling/tests'
# DOMAIN='/home/kb742/mev-adaptive-sampling/domain'
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --u_random_portion_gauss 1.0 --local_portion 0.0 --cross_portion 0.0 --early_stopping 1000
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --early_stopping 1000
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 20 --num_samples 10 --parents_portion 0. --early_stopping 1000
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 30 --num_samples 20 --parents_portion 0.0 --p_swap_max 0.8 --p_swap_min 0.1
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 30 --num_samples 20 --u_random_portion 1. --parents_portion 0. --early_stopping 1000

# TRANSACTIONS='./manualtests/jit_bigger_notype3_alpha'
# DOMAIN='./domain_notype3'
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --early_stopping 1000 --u_random_portion_gauss 0.4 --local_portion 0.3 --cross_portion 0.3
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 30 --num_samples 20 --parents_portion 0.0 --p_swap_max 0.8 --p_swap_min 0.1
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 30 --num_samples 20 --u_random_portion 1. --parents_portion 0.0

TRANSACTIONS='/home/kb742/mev-adaptive-sampling/clienttests_sushi/temp3'
# TRANSACTIONS='/home/kb742/mev-adaptive-sampling/clienttests_agg/12970075_12970100'
DOMAIN='/home/kb742/mev-adaptive-sampling/domain_client'
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --u_random_portion_gauss 1.0 --local_portion 0.0 --cross_portion 0.0
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --early_stopping 1000 --num_samples_gauss 10
python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 5 --num_samples 5 --parents_portion 0.0 --p_swap_max 0.5 --p_swap_min 0.1 --num_samples_gauss 20
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 30 --num_samples 20 --u_random_portion 1. --parents_portion 0. --early_stopping 1000
