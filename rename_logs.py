import os
import shutil

path_to_results = './artifacts_'
path_to_move = './artifacts'

for problem in os.listdir(path_to_results):
    subfolders = os.listdir(os.path.join(path_to_results, problem))
    for sf in subfolders:
        if sf in ['50iter_50nsamples_1.0random_0.0local_0.0_cross',
                  '20iter_10nsamples_0.2random_0.0parents_0.3p_swap',
                  '20iter_10nsamples_0.2random_0.0parents_0.5p_swap',
                  '20iter_10nsamples_0.2random_0.0parents_0.8p_swap',
                  '10iter_20nsamples_0.2random_0.0parents_0.5p_swap',
                  ]:
            curr_path = os.path.join(path_to_results, problem, sf)
            dest_path = os.path.join(path_to_move, problem, sf)
            print(f'moving {curr_path} to {dest_path}')
            shutil.copytree(curr_path, dest_path, dirs_exist_ok=True)

# for problem in os.listdir(path_to_move):
#     subfolders = os.listdir(os.path.join(path_to_move, problem))
#     for sf in subfolders:
#         if sf =='20iter_10nsamples_0.2random_0.0parents':
#             curr_path = os.path.join(path_to_move, problem, sf)
#             print(f'removing {curr_path}')
#             shutil.rmtree(curr_path)

