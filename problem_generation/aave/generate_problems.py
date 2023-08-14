import csv
import requests
import json
from collections import defaultdict
from web3 import Web3
import sys
import os
sys.path.append('..')
from templates import sushiswap, uniswapv3

aave_address = "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9"
ARCHIVE_NODE_URL = "http://localhost:8545"
v2_topics = ['0x4c209b5fc8ad50758f13e2e1088ba56a560dff690a1c6fef26394f4c03821c4f',
            '0xdccd412f0b1252819cb1fd330b93224ca42612892bb3f4f789976e6d81936496',
            '0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822']
v3_topics = ['0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67',
            '0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c',
            '0x70935338e69775456a85ddef226c395fb668b63fa0115f5f20610b388e6ca9c0',
            '0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde']
dex_topics = v2_topics + v3_topics

w3 = Web3(Web3.HTTPProvider(ARCHIVE_NODE_URL))
erc20_abi = json.loads(open('../abi/erc20_abi.json','r').read())
eth = Web3.toChecksumAddress("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2")


def get_aave_logs(block_number):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getLogs'
    data['params'] = [{"fromBlock": hex(block_number), "toBlock": hex(block_number), "address": aave_address}]
    data['id'] = 1
    r = requests.post(ARCHIVE_NODE_URL, json=data)
    response = json.loads(r.content)
    return response['result']

def get_tx_receipt(txhash):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getTransactionReceipt'
    data['params'] = [txhash]
    data['id'] = 1
    r = requests.post(ARCHIVE_NODE_URL, json=data)
    response = json.loads(r.content)
    return response['result']

def dex_involved(tx_receipt):
    for log in tx_receipt['logs']:
        if log['topics'][0] in dex_topics:
            return True
    return False

def get_aave_transactions(block_number):
    logs = get_aave_logs(block_number)
    candidate_transactions = set()
    res = []
    for log in logs:
        txhash = log['transactionHash']
        if txhash in candidate_transactions:
            continue
        tx_receipt = get_tx_receipt(txhash)
        if not dex_involved(tx_receipt):
            candidate_transactions.add(txhash)
            sender = tx_receipt['from']
            res.append('0,{},{}'.format(sender, txhash))
    return res

def get_decimals(token_addr):
    token_contract = w3.eth.contract(abi=erc20_abi, address=w3.toChecksumAddress(token_addr))
    return token_contract.functions.decimals().call()

infile = 'aave_interesting_blocks_from_logs.csv'

logsdict = csv.DictReader(open(infile), delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

problem_transactions = {}
problem_tokens = defaultdict(lambda: set())
problem_domain = {}

count = 0
#block,debtAsset,user_address,collateralAmount
for log in logsdict:
    block_number = int(log['block'])
    if block_number not in problem_transactions:
        problem_transactions[block_number] = get_aave_transactions(block_number)
        problem_domain[block_number] = []
    token_addr = Web3.toChecksumAddress(log['debtAsset'])
    user_addr = Web3.toChecksumAddress(log['userAddress'])
    template_variable = 'alpha' + str(len(problem_domain[block_number]) + 1)
    aave_template = '1,miner,Aave,0,liquidationCall,{},{},{},{},false'.format(
        eth, token_addr, user_addr, template_variable)
    problem_transactions[block_number].append(aave_template)
    problem_tokens[block_number].add(token_addr)
    problem_domain[block_number].append('{},{},{},{}'.format(template_variable,0,int(5*10**6),int(10**get_decimals(token_addr))))
    count += 1
    if count % 100 == 20:
        print(count)

for block_number in problem_tokens:
    for token_addr in problem_tokens[block_number]:
        template_variable = 'alpha' + str(len(problem_domain[block_number]) + 1)
        sushi_template = sushiswap.swap_template(template_variable, eth, token_addr)
        problem_domain[block_number].append('{},{},{},{}'.format(template_variable,0,2000,1))

        template_variables = ['alpha' + str(x) for x in range(len(problem_domain[block_number]) +1, len(problem_domain[block_number]) +4)]
        uni_templates = uniswapv3.swap_templates(template_variables, eth, token_addr)
        problem_transactions[block_number] = [sushi_template] + uni_templates + problem_transactions[block_number]
        for template_variable in template_variables:
            problem_domain[block_number].append('{},{},{},{}'.format(template_variable,0,2000,int(10**18)))

output_dir = '../../tests_liquidations/0xdeadc0de'

for block_number in problem_tokens:
    problem_dir = os.path.join(output_dir, str(block_number))
    os.makedirs(problem_dir , exist_ok=True)
    tx_filename = os.path.join(problem_dir, 'amm_reduced')
    domain_filename = os.path.join(problem_dir, 'domain')
    tx_file = open(tx_filename, 'w')
    tx_file.write('{},{}\n'.format(block_number, ','.join(problem_tokens[block_number])))
    tx_file.write('\n'.join(problem_transactions[block_number]))
    tx_file.close()
    domain_file = open(domain_filename, 'w')
    domain_file.write('\n'.join(problem_domain[block_number]))
    