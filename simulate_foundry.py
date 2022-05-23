
import requests
import json
import argparse
import logging
from copy import deepcopy
from contracts import utils
from contracts.uniswap import uniswap_router_contract, sushiswap_router_contract
from contracts.tokens import token_contracts, erc20_abi
from web3 import Web3
from collections import defaultdict
import time
import curlify

LARGE_NEGATIVE = -1e9
FORK_URL = 'http://localhost:8544'
ARCHIVE_NODE_URL = 'http://localhost:8545'
MINER_ADDRESS = '0x05E3bD644724652DD57246bae865d4A644151603'
USER_ADDRESS = '0x42bD55c6502E586d66020Ece3fdA53Cb73D73b6D'
USER1_ADDRESS = '0x183a57b895B8b4604E05C4886b26d9c710408c99'
USER2_ADDRESS = '0x76C2BD3aE2CA0F63A385259444cB11cf4Ec8e5ad'
USER3_ADDRESS = '0x15b118025c9d07522d1e0e6f0E15421741A44489'
MINER_KEY = '9a06d0fcf25eda537c6faa451d6b0129c386821d86062b57908d107ba022c4f3'
USER_KEY = 'd8d14136a29ac2d003983781f28b120fd144507f600876ad6479b92d747146ec'
USER1_KEY = '75534742a7f736a1e958c7b2bc790f45819b22eaf25d666cd71badc7dfd30663'
USER2_KEY = 'f0a05ef2d2bcc96062e9b404f453100ea7ae61d4c8acc0c09465cb82724e9659'
USER3_KEY = '36089ccdce092179345ee6df533f1be3ff66d280806aee8a3d0ff6289442185d'
KEYS = {MINER_ADDRESS: MINER_KEY, USER_ADDRESS: USER_KEY, USER1_ADDRESS: USER1_KEY, USER2_ADDRESS: USER2_KEY, USER3_ADDRESS: USER3_KEY}
MINER_CAPITAL = 1000*1e18

nonces = defaultdict(lambda: 0)


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

def get_balance(address, block):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getBalance'
    data['params'] = [address, block]
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
    data['method'] = 'anvil_reset'
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
def get_mev(block):
    # print(get_balance(MINER_ADDRESS, block))
    miner_balance = int(get_balance(MINER_ADDRESS, block)['result'], 16)
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

def parse_and_sign_basic_tx(elements, sender, w3):
    global nonces
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
        'nonce': nonces[sender],
        'chainId': 1,
    }
    tx = dynamic_tx
    # print(tx)
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=KEYS[sender])
    # print(signed_tx.rawTransaction.hex())
    return signed_tx.rawTransaction.hex()

def parse_and_sign_contract_tx(elements, sender, w3):
    global nonces
    to_address = elements[1]
    value = int(float(elements[2])*1e18) #given in eth, convert to wei
    func_name = elements[3]
    params = elements[4:]
    if to_address == 'UniswapV2Router02':
        contract = uniswap_router_contract
    elif to_address == 'SushiswapRouter':
        contract = sushiswap_router_contract
    else:
        contract = token_contracts[to_address]
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
        'nonce': nonces[sender],
        'chainId': 1,
    }
    tx = dynamic_tx
    # print(tx)
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=KEYS[sender])
    # print(signed_tx.rawTransaction.hex())
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
    data['params'] = [1]
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    return response


def disable_automining():
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'evm_setAutomine'
    data['params'] = [False]
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    return response


def query_forked_block(block_number):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getBlockByNumber'
    data['params'] = [block_number, True] # get full tx
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    return response

def simulate_tx(line, w3):
    global nonces
    line = line.replace('miner', MINER_ADDRESS)
    line = line.replace('user3', USER3_ADDRESS)
    line = line.replace('user2', USER2_ADDRESS)
    line = line.replace('user1', USER1_ADDRESS)
    line = line.replace('user', USER_ADDRESS)
    elements = line.strip().split(',')
    tx_type = elements[0]
    sender = elements[1]
    if tx_type == '0':
        # existing transaction
        serialized_tx = get_transaction(elements[2])['result']
        out = apply_transaction(serialized_tx)
        # print(out)
    elif tx_type == '1':
        # inserted transaction
        serialized_tx = parse_and_sign_contract_tx(elements[1:], sender, w3)
        out = apply_transaction(serialized_tx)
        # print(out)
        nonces[sender] += 1
    elif tx_type == '2':
        # inserted transaction
        serialized_tx = parse_and_sign_basic_tx(elements[1:], sender, w3)
        out = apply_transaction(serialized_tx)
        # print(out)
        nonces[sender] += 1

def simulate(lines, port_id):
    global nonces, FORK_URL
    FORK_URL = 'http://localhost:{}'.format(8544-port_id)
    w3 = Web3(Web3.HTTPProvider(FORK_URL))
    nonces = defaultdict(lambda : 0)
    bootstrap_line = lines[0].strip()
    bootstrap_block = int(bootstrap_line.split(',')[0]) - 1

    fork(bootstrap_block)
    disable_automining() #TODO : need to only do it once for the server

    for address in KEYS:    
        set_balance(address, int(MINER_CAPITAL))
    # set_balance(MINER_ADDRESS, int(MINER_CAPITAL))
    
    set_miner(MINER_ADDRESS)
    approved_tokens = bootstrap_line.split(',')[1:]
    for token in approved_tokens:
        if token.startswith('0x'):
            token_contracts[token] = w3.eth.contract(abi=erc20_abi, address=token)
    for token in approved_tokens:
        # simulate_tx('1,miner,{},0,approve,{},1000000000000000000000000000'.format(token, uniswap_router_contract.address)) #1e27
        simulate_tx('1,miner,{},0,approve,{},1000000000000000000000000000'.format(token, sushiswap_router_contract.address), w3) #1e27
    
    for line in lines[1:]:
        if line.startswith('#'):
            continue
        simulate_tx(line, w3)
    mine_block()
    # print(query_forked_block(hex(bootstrap_block+1)))
    # print(get_balance(MINER_ADDRESS, hex(bootstrap_block + 1)))
    # TODO : get the mined block, and make sure that it has the same number of mined tx as passed into the simulate method (+ any bootstrapping tx)
    mev  = get_mev(hex(bootstrap_block+1))
    return mev

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

    parser.add_argument(
        '-p', '--port',
        help="Id of one of the many backend client",
        required=False,
        default=24
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel, format='%(message)s')

    logger = logging.getLogger(__name__)

    data_f = open(args.file, 'r')
    port_id = int(args.port)
    mev = simulate(data_f.readlines(), port_id)
    print(mev)
