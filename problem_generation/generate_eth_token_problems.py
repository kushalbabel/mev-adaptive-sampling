import os

# filename = "eth_token_interesting_blocks.csv"
filename = "eth_token_interesting_blocks_from_logs.csv"
for line in open(filename, 'r').readlines():
    address = line.split(",")[0]
    block = line.split(",")[1].strip()
    cmd="python3 generate_client_problems.py -f /data/latest-data/uniswapv2-processed/{}.csv -sb {} -eb {} -o ../eth_token_tests_uniswapv2/{}".format(address, block, int(block)+1, address)
    os.system(cmd)
    # print(cmd)