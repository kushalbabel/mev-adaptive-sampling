import os
filename = "eth_token_interesting_blocks.csv"
for line in open(filename, 'r').readlines():
    address = line.split(",")[0]
    block = line.split(",")[1].strip()
    cmd="python3 generate_client_problems.py -f ../mev/data-scripts/latest-data/sushiswap-processed/{}.csv -sb {} -eb {} -o eth_token_tests/{}".format(address, block, int(block)+1, address)
    print(cmd)
    os.system(cmd)
