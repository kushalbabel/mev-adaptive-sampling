import csv, os
import pandas as pd
import logging
import json
import sys
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

aave_liquidation_logs = '/data/latest-data/aave_liquidation_logs.csv'

eth = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
# eth = "1097077688018008265106216665536940668749033598146"

logsdict = csv.DictReader(open(aave_liquidation_logs), delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

# logs sorted by block number and then transaction indices (all logs from same txhash are consecutive)

def topics_from_text(raw_text):
    return json.loads(raw_text.replace('\'', '\"'))

interesting_blocks = defaultdict(lambda : 0)
block_aave_events = defaultdict(lambda : [])
fout = open('aave_interesting_blocks_from_logs.csv', 'w')

#Interested in only Mint, Burn and Swap events
interested_topics = ['0xe413a321e8681d831f4dbccbca790d2952b56f977908e45be37335533e005286']
# start_block = 12400000 # uniswapv3 launch May 5'21
start_block = 12600000 # June 9'21
end_block = 15537351 # mev-boost launch


fout.write('block,debtAsset,userAddress,collateralAmount\n')

for log in logsdict:
    topics = topics_from_text(log['topics'])
    topics = log['topics'][1:-1]
    topics = topics.replace("'","").replace(" ", "").split(',')
    if topics[0] not in interested_topics:
        continue
    
    block_number = int(log['block_number'], 16)
    if block_number > end_block or block_number < start_block:
        continue

    # print("here")
    collateral_asset = "0x" + topics[1][-40:]
    debt_asset = "0x" + topics[2][-40:]
    user_address = "0x" + topics[3][-40:]
    # print(collateral_asset)
    if debt_asset != eth and collateral_asset != eth:
        continue
    

    data = log['data']
    data = data[2:] # strip 0x from hex

    debt_to_cover = int(str(data[:64]), 16)
    collateral_amount = int(str(data[64:128]), 16)

    if collateral_asset == eth:
        interesting_blocks[block_number] += collateral_amount
        block_aave_events[block_number].append('{},{},{},{}\n'.format(block_number, debt_asset,user_address, collateral_amount/1e18))
        

s = sorted(interesting_blocks.items(), key=lambda a: -a[1])
for item in s:
    block_number = item[0]
    for event in block_aave_events[block_number]:
        fout.write(event)

logger.info("Done...")

#log_index,transaction_hash,address,data,topics,block_number
#0x38,0x9ef1c383398cee036a85ca024db051b7a1c8bc692e703fe1b230da9e2305eb4a,0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9,0x0000000000000000000000000000000000000000000000000000000000012241000000000000000000000000000000000000000000000000f71a83c58b15b4eb0000000000000000000000007a512a3cf68df453ec76d487e3eaffecd74d68870000000000000000000000000000000000000000000000000000000000000000,"['0xe413a321e8681d831f4dbccbca790d2952b56f977908e45be37335533e005286', '0x0000000000000000000000006b175474e89094c44da98b954eedeac495271d0f', '0x0000000000000000000000002260fac5e5542a773aa44fbcfedf7c193bc2c599', '0x000000000000000000000000a53fe221bd861f75907d8fd496db1c70779721aa']",0xaf0943