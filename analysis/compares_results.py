import yaml
f1 = '/home/gid-javaheripim/mev-adaptive-sampling/artifacts_smooth/info_summary.yaml'
f2 = '/home/gid-javaheripim/mev-adaptive-sampling/artifacts_smooth_uniswapv2/info_summary.yaml'

with open(f1, "r") as stream:
    r1 = yaml.safe_load(stream)

with open(f2, "r") as stream:
    r2 = yaml.safe_load(stream)

counter = 0
for problem in r1:
    counter += 1
    v1 = r1[problem]
    v2 = r2[problem]
    if v1 < v2:
        print(problem, v1, v2)
print(counter)
