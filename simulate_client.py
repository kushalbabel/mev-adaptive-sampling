import requests
import json
import argparse
import logging
from uniswapv2 import UniswapV2
from copy import deepcopy
from contracts import utils
from contracts.uniswap import uniswap_router_contract
from web3 import Web3

LARGE_NEGATIVE = -1e9
FORK_URL = 'http://localhost:8546'
ARCHIVE_NODE_URL = 'http://localhost:8545'
MINER_ADDRESS = '0x05E3bD644724652DD57246bae865d4A644151603'
MINER_KEY = '9a06d0fcf25eda537c6faa451d6b0129c386821d86062b57908d107ba022c4f3'
MINER_CAPITAL = 1000*1e18

miner_nonce = 0
w3 = Web3(Web3.HTTPProvider(FORK_URL))

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

def parse_and_sign_basic_tx(elements):
    global miner_nonce
    to_address = elements[1]
    value = int(float(elements[2])*1e18) #given in eth, convert to wei
    #TODO : dynamic vs normal tx, take care at London boundary, or always use old after fetching basefees
    dynamic_tx = {
        'to': to_address,
        'value': value,
        'gas': 15000000,
        # 'gasPrice': 76778040978,
        'maxFeePerGas': 146778040978,
        'maxPriorityFeePerGas':1000,
        'nonce': miner_nonce,
        'chainId': 1,
    }
    miner_nonce += 1 #TODO: increment at the right place, taking care of tx failures
    tx = dynamic_tx
    print(tx)
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=MINER_KEY)
    print(signed_tx.rawTransaction.hex())
    return signed_tx.rawTransaction.hex()

def parse_and_sign_contract_tx(elements):
    global miner_nonce
    to_address = elements[1]
    value = int(float(elements[2])*1e18) #given in eth, convert to wei
    func_name = elements[3]
    params = elements[4:]
    if to_address == 'UniswapV2Router02':
        contract = uniswap_router_contract
        calldata = utils.encode_function_call1(contract, func_name, params)
        #TODO : dynamic vs normal tx, take care at London boundary, or always use old after fetching basefees
        dynamic_tx = {
            'to': contract.address,
            'value': value,
            'data': calldata,
            'gas': 15000000,
            # 'gasPrice': 76778040978,
            'maxFeePerGas': 146778040978,
            'maxPriorityFeePerGas':1000,
            'nonce': miner_nonce,
            'chainId': 1,
        }
        miner_nonce += 1 #TODO: increment at the right place, taking care of tx failures
        tx = dynamic_tx
        print(tx)
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=MINER_KEY)
        print(signed_tx.rawTransaction.hex())
        return signed_tx.rawTransaction.hex()


def set_miner(address):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'hardhat_setCoinbase'
    data['params'] = [address]
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    return response

def mine_block():
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'evm_mine'
    data['params'] = []
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    return response


def simulate(lines):
    bootstrap_line = lines[0].strip()
    bootstrap_block = int(bootstrap_line) - 1
    #setup
    fork(bootstrap_block)
    set_balance(MINER_ADDRESS, int(MINER_CAPITAL))
    set_miner(MINER_ADDRESS)

    #simulate transactions
    for line in lines[1:]:
        if line.startswith('#'):
            #TODO remove for performance in prod
            continue
        elements = line.strip().split(',')
        tx_type = elements[0]
        if tx_type == '0':
            # existing transaction
            serialized_tx = get_transaction(elements[2])['result']
            print(apply_transaction(serialized_tx))
        elif tx_type == '1':
            # inserted transaction
            serialized_tx = parse_and_sign_contract_tx(elements[1:])
            print(apply_transaction(serialized_tx))
        elif tx_type == '2':
            # inserted transaction
            serialized_tx = parse_and_sign_basic_tx(elements[1:])
            print(apply_transaction(serialized_tx))
    print(mine_block())
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
