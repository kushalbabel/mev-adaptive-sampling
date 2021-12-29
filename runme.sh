# python optimize.py -t toy_problem -d domain --num_samples 20 --n_iter 10 --plot_contour --early_stopping 5 --u_random_portion 0.3 --local_portion 0.35 --cross_portion 0.35
# python optimize.py -t /home/kb742/mev-adaptive-sampling/smallertests -d /home/kb742/mev-adaptive-sampling/domain \
#                 --reorder --n_iter 20 --num_samples 10 --parents_portion 0.
python optimize.py -t /home/kb742/mev-adaptive-sampling/smallertests -d /home/kb742/mev-adaptive-sampling/domain \
                --reorder --n_iter 20 --num_samples 10 --u_random_portion 1. --parents_portion 0.