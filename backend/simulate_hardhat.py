from unicodedata import decimal
import requests
import json
import argparse
import sys
import pandas as pd
import logging
from copy import deepcopy
from contracts import utils
from contracts.uniswap import uniswap_router_contract, sushiswap_router_contract
from contracts.tokens import token_contracts, erc20_abi
from web3 import Web3
from collections import defaultdict
import logging
import time
from datetime import datetime
from utils import get_price

simlogger = logging.getLogger(__name__)
sim_log_handler = logging.FileHandler('output.log')
simlogger.addHandler(sim_log_handler)
simlogger.setLevel(logging.DEBUG)
simlogger.propagate = False

LARGE_NEGATIVE = -1e9
BLOCKREWARD = 2
weth = '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'
FORK_URL = 'http://localhost:8547'
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
prices = dict()
decimals = dict()
dexes = dict()

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

def get_decimals(token_addr):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_call'
    function_selector = "0x313ce567000000000000000000000000"
    calldata = function_selector
    data["params"] = [{"to": token_addr, "data":calldata}, "latest"]
    # now = datetime.now()
    data['id'] = 1
    r = requests.post(ARCHIVE_NODE_URL, json=data)
    response = json.loads(r.content)
    return int(response["result"], 16)

def get_reserves(exchange_addr):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_call'
    function_selector = "0x0902f1ac000000000000000000000000"
    calldata = function_selector
    data["params"] = [{"to": exchange_addr, "data":calldata}, "latest"]
    # now = datetime.now()
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    result = response["result"]
    reserve0 = result[2:66]
    reserve1 = result[66:130]
    ts = result[130:]
    print("Reserve TS", int(ts, 16))
    return reserve0, reserve1

def getAmountOutv1(token_addr, exchange_addr, router_contract, in_amount):
    reserve0, reserve1 = get_reserves(exchange_addr)
    print("reserve0", int(reserve0, 16), "reserve1", int(reserve1, 16))
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_call'
    if int(token_addr, 16) < int(weth, 16):
        calldata = utils.encode_function_call1(router_contract, 'getAmountOut', [in_amount, int(reserve0, 16), int(reserve1, 16)])
    else:
        calldata = utils.encode_function_call1(router_contract, 'getAmountOut', [in_amount, int(reserve1, 16), int(reserve0, 16)])
    data["params"] = [{"to": router_contract.address, "data":calldata}, "latest"]
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    if 'result' in response:
        return int(response['result'], 16)
    else:
        # TODO log response
        return 0

def getAmountOutv2(token_addr, exchange_addr, router_contract, in_amount):
    reserve0, reserve1 = get_reserves(exchange_addr)
    print("reserve0", int(reserve0, 16), "reserve1", int(reserve1, 16))
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_call'
    calldata = utils.encode_function_call1(router_contract, 'getAmountsOut', [in_amount, '[{}-0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2]'.format(token_addr)])
    data["params"] = [{"to": router_contract.address, "data":calldata}, "latest"]
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    if 'result' in response:
        return int(response['result'][-64:], 16)
    else:
        # TODO log response
        return 0


# in eth
def get_mev():
    miner_balance = int(get_balance(MINER_ADDRESS)['result'], 16)
    mev = miner_balance - MINER_CAPITAL
    return mev/1e18

# in eth
def get_mev_cex(remaining_balances):
    ret = 0
    eth_balance = get_mev()
    ret += eth_balance
    for token in remaining_balances:
        token_balance = remaining_balances[token] / (10**decimals[token])
        ret += token_balance* prices[token] / prices['eth']
    return ret

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
        'maxFeePerGas': 14677804097800,
        'maxPriorityFeePerGas':1000,
        'nonce': nonces[sender],
        'chainId': 1,
    }
    tx = dynamic_tx
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=KEYS[sender])
    # print(signed_tx.rawTransaction.hex())
    return signed_tx.rawTransaction.hex()

def parse_and_sign_contract_tx(elements, sender, w3):
    global nonces
    to_address = elements[1]
    value = int(float(elements[2])*1e18) #given in eth, convert to wei
    func_name = elements[3]
    params = elements[4:]
    # clean up and make the to_addr uniformly handled
    if to_address == 'UniswapV2Router':
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
        'maxFeePerGas': 14677804097800,
        'maxPriorityFeePerGas':1000,
        'nonce': nonces[sender],
        'chainId': 1,
    }
    tx = dynamic_tx
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
    data['params'] = []
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

def get_token_balance(user_addr, token_addr):
    # contract = token_contracts[token_addr]
    # balance = contract.functions.balanceOf(user_addr).call()
    # return balance
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_call'
    function_selector = "0x70a08231000000000000000000000000"
    calldata = function_selector + user_addr.replace("0x","")
    data["params"] = [{"to": token_addr, "data":calldata}, "latest"]
    now = datetime.now()
    data['id'] = time.mktime(now.timetuple())*1e6 + now.microsecond
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    simlogger.debug("[REQUEST] %s", data)
    simlogger.debug("[RESPONSE] %s", response)
    balance = int(response['result'], 16)
    return balance

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

# bootstrap_line : the first line of the problem
def setup(bootstrap_line):
    bootstrap_line = bootstrap_line.strip()
    global prices, decimals, dexes
    w3 = Web3(Web3.HTTPProvider(ARCHIVE_NODE_URL))
    approved_tokens = bootstrap_line.split(',')[1:]
    for token in approved_tokens:
        if token.startswith('0x'):
            token_contracts[token] = w3.eth.contract(abi=erc20_abi, address=token)
    df = pd.read_csv('/data/latest-data/uniswapv2_pairs.csv')
    dexes = dict()
    sushiswap = '0xc0aee478e3658e2610c5f7a4a2e1777ce9e4f2ac'
    uniswapv2 = '0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f'
    for token in approved_tokens:
        dexes[token] = dict()
        if int(weth, 16) > int(token, 16):
            uniswapv2_df = df[(df['exchange'] == uniswapv2) & ((df['token0'] == token.lower()) & (df['token1'] == weth)) ]
            sushiswap_df = df[(df['exchange'] == sushiswap) & ((df['token0'] == token.lower()) & (df['token1'] == weth))]
        else:
            uniswapv2_df = df[(df['exchange'] == uniswapv2) & ((df['token1'] == token.lower()) & (df['token0'] == weth)) ]
            sushiswap_df = df[(df['exchange'] == sushiswap) & ((df['token1'] == token.lower()) & (df['token0'] == weth))]
        if len(uniswapv2_df) > 0:
            dexes[token]['UniswapV2'] = uniswapv2_df.iloc[0].pair
        if len(sushiswap_df) > 0:
            dexes[token]['Sushiswap'] = sushiswap_df.iloc[0].pair
    prices = dict()
    decimals = dict()
    
    bootstrap_block = int(bootstrap_line.split(',')[0]) - 1
    involved_tokens = approved_tokens
    prices['eth'] = get_price(bootstrap_block, 'eth')
    for token in involved_tokens:
        decimals[token] = get_decimals(token)
    try:
        for token in involved_tokens:
            prices[token] = get_price(bootstrap_block, token)
    except:
        pass


def simulate(lines, port_id, best=False, logfile=None, settlement='max'):
    global nonces, FORK_URL
    FORK_URL = 'http://localhost:{}'.format(8547+port_id)
    w3 = Web3(Web3.HTTPProvider(FORK_URL))
    nonces = defaultdict(lambda : 0)
    bootstrap_line = lines[0].strip()
    simlogger.debug("[ %s ]", bootstrap_line)
    bootstrap_block = int(bootstrap_line.split(',')[0]) - 1
    approved_tokens = bootstrap_line.split(',')[1:]
    

    # Prepare state
    fork(bootstrap_block)
    for address in KEYS:
        set_balance(address, int(MINER_CAPITAL))
    set_miner(MINER_ADDRESS)
    
    # Preparation transactions
    for token in approved_tokens:
        approve_tx = '1,miner,{},0,approve,{},1000000000000000000000000000'.format(token, sushiswap_router_contract.address)
        simulate_tx(approve_tx, w3) #1e27
        approve_tx = '1,miner,{},0,approve,{},1000000000000000000000000000'.format(token, uniswap_router_contract.address)
        simulate_tx(approve_tx, w3) #1e27
    

    # Execute transactions
    for line in lines[1:]:
        if line.startswith('#'):
            continue
        simulate_tx(line, w3)
    
    # Mine the transactions
    mine_result = mine_block()
    if 'error' in mine_result:
        simlogger.debug(mine_result['error'])
        return None


    # get remaining balances
    remaining_balances = {}
    for token in approved_tokens:
        token_addr = token_contracts[token].address
        balance = None
        debug_counter = 0
        while balance is None:
            try:
                balance = get_token_balance(MINER_ADDRESS, token_addr)
            except:
                balance = None
                debug_counter += 1
                simlogger.debug("[COUNTER] %d", debug_counter)
        remaining_balances[token_addr] = balance

    mev = 0

    if settlement == 'cex' or settlement == 'max':
        # view only calculation
        try:
            mev = max(mev, BLOCKREWARD + get_mev_cex(remaining_balances))  # blockreward for parity with dex
        except KeyError:
            simlogger.debug("Not listed on binance")
    if settlement == 'dex' or settlement == 'max':
        # settle on dex, calculation changes STATE!!
        for token_addr in remaining_balances:
            remaining_balance = remaining_balances[token_addr]
            if remaining_balance <= 0:
                continue
            uniswapv2_out_amount = getAmountOutv2(token_addr, dexes[token_addr]['UniswapV2'], uniswap_router_contract, remaining_balance)
            sushiswap_out_amount = getAmountOutv2(token_addr, dexes[token_addr]['Sushiswap'], sushiswap_router_contract, remaining_balance)
            if sushiswap_out_amount > uniswapv2_out_amount:
                automatic_tx = '1,miner,SushiswapRouter,0,swapExactTokensForETH,{},0,[{}-0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2],miner,1800000000'.format(remaining_balance, token_addr)
            else:
                automatic_tx = '1,miner,UniswapV2Router,0,swapExactTokensForETH,{},0,[{}-0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2],miner,1800000000'.format(remaining_balance, token_addr)
            simulate_tx(automatic_tx, w3)
        # mine the block to execute automatic_tx
        mine_block()
        mev = max(mev, get_mev())

    # store the best sample
    if best:
        best_sample = []
        best_sample += lines
        for token_addr in remaining_balances:
            best_sample.append('#balance {}:{},{}'.format(token_addr, remaining_balances[token_addr]/(10**decimals[token_addr]),decimals[token_addr]))
        with open(logfile, 'w') as flog:
            for tx in best_sample:
                flog.write('{}\n'.format(tx.strip()))

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
        default=-1
    )

    parser.add_argument(
        '-s', '--settlement',
        help="cex/dex/max",
        required=False,
        default='max'
    )

    args = parser.parse_args()
    
    data_f = open(args.file, 'r')
    port_id = int(args.port)
    lines = data_f.readlines()
    print("setting up...", lines[0])
    setup(lines[0])
    print("simulating...")
    mev = simulate(lines, port_id, False, '', args.settlement)
    print(mev)
