TRANSACTIONS='/home/kb742/mev-adaptive-sampling/smallertests'
DOMAIN='/home/kb742/mev-adaptive-sampling/domain'
python optimize.py -t $TRANSACTIONS -d $DOMAIN --u_random_portion_gauss 1.0 --local_portion 0.0 --cross_portion 0.0
python optimize.py -t $TRANSACTIONS -d $DOMAIN --early_stopping 1000
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 20 --num_samples 10 --parents_portion 0. --early_stopping 1000
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 10 --num_samples 20 --parents_portion 0. --early_stopping 1000
# python optimize.py -t $TRANSACTIONS -d $DOMAIN --reorder --n_iter 20 --num_samples 10 --u_random_portion 1. --parents_portion 0. --early_stopping 1000