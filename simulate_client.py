import requests
import json
import argparse
import logging
from uniswapv2 import UniswapV2
from copy import deepcopy

LARGE_NEGATIVE = -1e9
FORK_URL = 'http://localhost:8546'
ARCHIVE_NODE_URL = 'http://localhost:8545'
MINER_ADDRESS = '0x05E3bD644724652DD57246bae865d4A644151603'
MINER_CAPITAL = 1000*1e18

def query_block(block_number):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getBlockByNumber'
    data['params'] = [block_number, True] # get full tx
    data['id'] = block_number + 1000000
    r = requests.post(ARCHIVE_NODE_URL, json=data)
    response = json.loads(r.content)
    return response

def get_erigon_balance(address):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getBalance'
    data['params'] = [address, 14136754]
    data['id'] = 1
    r = requests.post(ARCHIVE_NODE_URL, json=data)
    response = json.loads(r.content)
    return response

def get_balance(address):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getBalance'
    data['params'] = [address]
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    return response


def get_currentBlock():
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_blockNumber'
    data['params'] = []
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    return response

# balance in wei
def set_balance(address, balance):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'hardhat_setBalance'
    data['params'] = [address, hex(balance)]
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    return response

def fork(bno):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'hardhat_reset'
    data['params'] = [{
        "forking": {
            "jsonRpcUrl": ARCHIVE_NODE_URL,
            "blockNumber": bno
        }
    }]
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    return response

def info(amm):
    print(dict(amm.config()))
    print('Supply',amm.supply)

def generate_transaction(tx_type, params):
    if tx_type == '1':
        format_string = '{} adds {} {} and {} {} of liquidity'
    elif tx_type == '2':
        format_string = '{} removes {} {} and {} {} of liquidity'
    elif tx_type == '3':
        format_string = '{} swaps for {} by providing {} {} and {} {} with change {} fee {}'
    elif tx_type == '4':
        format_string = '{} redeems {} fraction of liquidity from {} and {}'
    return format_string.format(*params)

# in eth
def get_mev():
    miner_balance = int(get_balance(MINER_ADDRESS)['result'], 16)
    mev = miner_balance - MINER_CAPITAL
    return mev/1e18

def get_transaction(tx_hash):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getRawTransactionByHash'
    data['params'] = [tx_hash]
    data['id'] = 1
    r = requests.post(ARCHIVE_NODE_URL, json=data)
    response = json.loads(r.content)
    return response

def apply_transaction(serialized_tx):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_sendRawTransaction'
    data['params'] = [serialized_tx]
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    return response

def parse_basic_transaction(elements):
    raise NotImplementedError

def parse_contract_transaction(elements):
    raise NotImplementedError


def simulate(lines):
    bootstrap_line = lines[0].strip()
    bootstrap_block = int(bootstrap_line) - 1
    #setup
    fork(bootstrap_block)
    set_balance(MINER_ADDRESS, int(MINER_CAPITAL))
    #simulate transactions
    for line in lines[1:]:
        elements = line.strip().split(',')
        tx_type = elements[0]
        if tx_type == '0':
            # existing transaction
            serialized_tx = get_transaction(elements[1])['result']
            print(apply_transaction(serialized_tx))
        elif tx_type == '1':
            # inserted transaction
            serialized_tx = parse_contract_transaction(elements[1:])
            print(apply_transaction(serialized_tx))
        elif tx_type == '2':
            # inserted transaction
            serialized_tx = parse_basic_transaction(elements[1:])
            print(apply_transaction(serialized_tx))
    return get_mev()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Simulation on a mainnet fork')

    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
        default=logging.WARNING
    )

    parser.add_argument(
        '-f', '--file',
        help="Input File path",
        required=True
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel, format='%(message)s')

    logger = logging.getLogger(__name__)

    data_f = open(args.file, 'r')
    mev = simulate(data_f.readlines())
    print(mev)
