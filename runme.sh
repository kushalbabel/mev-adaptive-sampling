TRANSACTIONS='/home/kb742/mev-adaptive-sampling/tests'
DOMAIN='/home/kb742/mev-adaptive-sampling/domain'
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --u_random_portion_gauss 1.0 --local_portion 0.0 --cross_portion 0.0
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --early_stopping 1000
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 20 --num_samples 10 --parents_portion 0. --early_stopping 1000
python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 30 --num_samples 20 --parents_portion 0.0 --p_swap_max 1.0 --p_swap_min 0.1
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 20 --num_samples 10 --u_random_portion 1. --parents_portion 0. --early_stopping 1000