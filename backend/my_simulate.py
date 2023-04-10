from unicodedata import decimal
import requests
import json
import argparse
import sys
import logging
from copy import deepcopy
from contracts import utils
from contracts.uniswap import uniswap_router_contract, sushiswap_router_contract, uniswapv3_router_contract, uniswapv3_quoter_abi, temp_abi
from contracts.tokens import token_contracts, erc20_abi, weth_abi
from web3 import Web3
from collections import defaultdict
import logging
import time
from datetime import datetime
from utils import get_price
import rlp
import eth_abi
from eth_utils import keccak, to_checksum_address, to_bytes

simlogger = logging.getLogger(__name__)
sim_log_handler = logging.FileHandler('output.log')
simlogger.addHandler(sim_log_handler)
simlogger.setLevel(logging.DEBUG)
#simlogger.setLevel(60)
simlogger.propagate = False

LARGE_NEGATIVE = -1e9
BLOCKREWARD = 2
WETH = '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'
FORK_URL = 'http://localhost:8547'
ARCHIVE_NODE_URL = 'http://localhost:8545'
MINER_ADDRESS = '0x05E3bD644724652DD57246bae865d4A644151603'
MINER_KEY = '9a06d0fcf25eda537c6faa451d6b0129c386821d86062b57908d107ba022c4f3'
KEYS = {MINER_ADDRESS: MINER_KEY}
MINER_CAPITAL = 2000*1e18
MY_SIMULATE_TX = True
MY_SETUP_LINES = 1

nonces = defaultdict(lambda: 0)
deployed_contract_addr = "0x"
prices = dict()
decimals = dict()

total_time = 0

def query_block(block_number):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getBlockByNumber'
    data['params'] = [block_number, True] # get full tx
    data['id'] = block_number + 1000000
    global total_time
    current = time.time()
    r = requests.post(ARCHIVE_NODE_URL, json=data)
    response = json.loads(r.content)
    total_time += time.time() - current
    return response

def get_erigon_balance(address):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getBalance'
    data['params'] = [address, 14136754]
    data['id'] = 1
    global total_time
    current = time.time()
    r = requests.post(ARCHIVE_NODE_URL, json=data)
    response = json.loads(r.content)
    total_time += time.time() - current
    return response

def get_balance(address):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getBalance'
    data['params'] = [address]
    data['id'] = 1
    global total_time
    current = time.time()
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    total_time += time.time() - current
    return response


def get_currentBlock():
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_blockNumber'
    data['params'] = []
    data['id'] = 1
    global total_time
    current = time.time()
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    total_time += time.time() - current
    return response

# balance in wei
def set_balance(address, balance):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'hardhat_setBalance'
    data['params'] = [address, hex(balance)]
    data['id'] = 1
    global total_time
    current = time.time()
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    total_time += time.time() - current
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
    global total_time
    current = time.time()
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    fork_time = time.time() - current
    print("fork time: ", fork_time)
    total_time += fork_time
    return response

def myFork(bno):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'hardhat_myFork'
    data['params'] = [{
        "forking": {
            "jsonRpcUrl": ARCHIVE_NODE_URL,
            "blockNumber": bno
        }
    }]
    data['id'] = 1
    global total_time
    current = time.time()
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    fork_time = time.time() - current
    print("my fork time: ", fork_time)
    total_time += fork_time
    return response

def mySnapshot():
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'hardhat_mySnapshot'
    data['params'] = []
    data['id'] = 1
    global total_time
    current = time.time()
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    fork_time = time.time() - current
    print("my snapshot time: ", fork_time)
    total_time += fork_time
    return response

def myReset():
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'hardhat_myReset'
    data['params'] = []
    data['id'] = 1
    global total_time
    current = time.time()
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    fork_time = time.time() - current
    print("my reset time: ", fork_time)
    total_time += fork_time
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
    global total_time
    current = time.time()
    r = requests.post(ARCHIVE_NODE_URL, json=data)
    response = json.loads(r.content)
    total_time += time.time() - current
    return int(response["result"], 16)

def get_code(contract_addr):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getCode'
    data["params"] = [contract_addr, "latest"]
    # now = datetime.now()
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    # return int(response["result"], 16)
    return response     

def getAmountOutv2(token_addr, router_contract, in_amount):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_call'
    calldata = utils.encode_function_call1(router_contract, 'getAmountsOut', [in_amount, '[{}-{}]'.format(token_addr, WETH)])
    data["params"] = [{"to": router_contract.address, "data":calldata}, "latest"]
    data['id'] = 1
    global total_time
    current = time.time()
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    total_time += time.time() - current
    if 'result' in response:
        return int(response['result'][-64:], 16)
    else:
        # TODO log response
        return 0

def get_best_fee_pool(token_addr, remaining_balance, w3):
    fee_options = [500, 3000, 10000]
    quoter_contract = w3.eth.contract(abi=uniswapv3_quoter_abi, address='0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6')
    max_quote = 0
    best_fee = fee_options[0]
    for fee in fee_options:
        try:
            amount = quoter_contract.functions.quoteExactInputSingle(token_addr, WETH, fee, remaining_balance,0).call()
            if amount > max_quote:
                best_fee = fee
                max_quote = amount
        except ValueError:
            pass
    return best_fee, max_quote


# in eth
def get_mev():
    miner_balance = int(get_balance(MINER_ADDRESS)['result'], 16)
    miner_balance += get_token_balance(MINER_ADDRESS, WETH)
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

def get_mev_dex(token_addr, remaining_balance, w3, involved_dexes):
    if 'uniswapv2' in involved_dexes:
        uniswapv2_out_amount = getAmountOutv2(token_addr, uniswap_router_contract, remaining_balance)
    else:
        uniswapv2_out_amount = 0
    if 'sushiswap' in involved_dexes:
        sushiswap_out_amount = getAmountOutv2(token_addr, sushiswap_router_contract, remaining_balance)
    else:
        sushiswap_out_amount = 0
    if 'uniswapv3' in involved_dexes:
        fee_pool, uniswapv3_out_amount = get_best_fee_pool(token_addr, remaining_balance, w3)
    else:
        uniswapv3_out_amount = 0
    return get_mev() + max(uniswapv3_out_amount, uniswapv2_out_amount, sushiswap_out_amount) / 1e18

def get_transaction(tx_hash):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getRawTransactionByHash'
    data['params'] = [tx_hash]
    data['id'] = 1
    global total_time
    current = time.time()
    r = requests.post(ARCHIVE_NODE_URL, json=data)
    response = json.loads(r.content)
    total_time += time.time() - current
    return response

def get_contract_address(sender: str, nonce: int) -> str:
    sender_bytes = to_bytes(hexstr=sender)
    raw = rlp.encode([sender_bytes, nonce])
    h = keccak(raw)
    address_bytes = h[12:]
    return to_checksum_address(address_bytes)

def apply_transaction(serialized_tx):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_sendRawTransaction'
    if MY_SIMULATE_TX:
        data['method'] = 'eth_mySimulateTx'
    data['params'] = [serialized_tx]
    data['id'] = 1
    global total_time
    current = time.time()
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    time_cost = time.time() - current
    total_time += time_cost
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
    return tx.hex()
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
    elif to_address == 'UniswapV3Router':
        contract = uniswapv3_router_contract
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

def get_bytecode_tx(bytecode, sender, w3):
    global nonces
    dynamic_tx = {
        'data': bytecode,
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

def get_nonce(address):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getTransactionCount'
    data['params'] = [address, "latest"]
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    return response

def set_miner(address):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'hardhat_setCoinbase'
    data['params'] = [address]
    data['id'] = 1
    global total_time
    current = time.time()
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    total_time += time.time() - current
    return response

def mine_block():
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'evm_mine'
    if MY_SIMULATE_TX:
        data['method'] = 'evm_myMine'
    data['params'] = []
    data['id'] = 1
    global total_time
    current = time.time()
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    total_time += time.time() - current
    return response

def query_forked_block(block_number):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getBlockByNumber'
    data['params'] = [block_number, True] # get full tx
    data['id'] = 1
    global total_time
    current = time.time()
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    total_time += time.time() - current
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
    global total_time
    current = time.time()
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    total_time += time.time() - current
    simlogger.debug("[REQUEST] %s", data)
    simlogger.debug("[RESPONSE] %s", response)
    balance = int(response['result'], 16)
    return balance

def simulate_tx(line, w3):
    global nonces
    line = line.replace('miner', MINER_ADDRESS)
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
    elif tx_type == '3':
        # inserted transaction
        bytecode = elements[2]
        serialized_tx = get_bytecode_tx(bytecode, sender, w3)
        out = apply_transaction(serialized_tx)
        print(out)
        contract_address = get_contract_address(sender, nonces[sender])
        nonces[sender] += 1
        print(contract_address)
        return contract_address

# bootstrap_line : the first line of the problem
def setup(lines, port_id, involved_dexes):
    bootstrap_line = lines[0].strip()
    approved_tokens = bootstrap_line.split(',')[1:]
    
    global prices, decimals
    w3 = Web3(Web3.HTTPProvider(ARCHIVE_NODE_URL))
    approved_tokens = bootstrap_line.split(',')[1:]
    for token in approved_tokens:
        if token.startswith('0x'):
            token_contracts[token] = w3.eth.contract(abi=erc20_abi, address=token)
    token_contracts[WETH] = w3.eth.contract(abi=weth_abi, address=WETH)
    prices = dict()
    decimals = dict()
    
    bootstrap_block = int(bootstrap_line.split(',')[0]) - 1
    involved_tokens = approved_tokens
    try:
        #TODO : update binance, then shouldnt run into exception, and remove try
        prices['eth'] = get_price(bootstrap_block, 'eth')
    except:
        pass
    for token in involved_tokens:
        decimals[token] = get_decimals(token)
    try:
        for token in involved_tokens:
            prices[token] = get_price(bootstrap_block, token)
    except:
        pass
    
    global FORK_URL
    FORK_URL = 'http://localhost:{}'.format(8547+port_id)
    myFork(bootstrap_block)
    
    for address in KEYS:
        set_balance(address, int(MINER_CAPITAL))
    set_miner(MINER_ADDRESS)

    if MY_SIMULATE_TX:
        global nonces
        nonces = defaultdict(lambda : 0)
        w3 = Web3(Web3.HTTPProvider(FORK_URL))
	
        # deploy contract
        # contract_bytecode = json.load(open('/home/kb742/v3-periphery/artifacts/contracts/NonfungiblePositionManager.sol/NonfungiblePositionManager.json'))["bytecode"]
        contract_bytecode = json.load(open('/home/kb742/v3-periphery/artifacts/contracts/temp.sol/User.json'))["bytecode"]
        constructor_params_encoded = b'' + eth_abi.encode_single('int', 1)
        global deployed_contract_addr
        deployed_contract_addr = simulate_tx("3,miner,{}".format(contract_bytecode + constructor_params_encoded.hex()), w3)
        token_contracts[deployed_contract_addr] = w3.eth.contract(abi=temp_abi, address=deployed_contract_addr)

        # Preparation transactions
        for token in approved_tokens:
            if 'sushiswap' in involved_dexes:
                approve_tx = '1,miner,{},0,approve,{},1000000000000000000000000000'.format(token, sushiswap_router_contract.address)
                simulate_tx(approve_tx, w3) #1e27
            if 'uniswapv2' in involved_dexes:
                approve_tx = '1,miner,{},0,approve,{},1000000000000000000000000000'.format(token, uniswap_router_contract.address)
                simulate_tx(approve_tx, w3) #1e27
            if 'uniswapv3' in involved_dexes:
                approve_tx = '1,miner,{},0,approve,{},1000000000000000000000000000'.format(token, uniswapv3_router_contract.address)
                simulate_tx(approve_tx, w3) #1e27
                approve_tx = '1,miner,{},0,approve,{},1000000000000000000000000000'.format(WETH, uniswapv3_router_contract.address)
                simulate_tx(approve_tx, w3) #1e27
                approve_tx = '1,miner,{},{},deposit'.format(WETH, int(MINER_CAPITAL/2/1e18))
                simulate_tx(approve_tx, w3) #1e27
    
    for line in lines[1:MY_SETUP_LINES]:
        if line.startswith('#'):
            continue
        simulate_tx(line, w3)
    
    mySnapshot()

    global snapshot_nonces
    snapshot_nonces = nonces.copy()


def simulate(lines, port_id, involved_dexes, best=False, logfile=None, settlement='max'):
    # Note that nonces need to sync up with the snapshot!
    global deployed_contract_addr
    global nonces, snapshot_nonces
    nonces = snapshot_nonces.copy()

    w3 = Web3(Web3.HTTPProvider(FORK_URL))
    bootstrap_line = lines[0].strip()
    simlogger.debug("[ %s ]", bootstrap_line)
    bootstrap_block = int(bootstrap_line.split(',')[0]) - 1
    approved_tokens = bootstrap_line.split(',')[1:]
    
    # Prepare state
    myReset()
    
    if not MY_SIMULATE_TX:
        w3 = Web3(Web3.HTTPProvider(FORK_URL))
    	
	# deploy contract
    	# contract_bytecode = json.load(open('/home/kb742/v3-periphery/artifacts/contracts/NonfungiblePositionManager.sol/NonfungiblePositionManager.json'))["bytecode"]
        contract_bytecode = json.load(open('/home/kb742/v3-periphery/artifacts/contracts/temp.sol/User.json'))["bytecode"]
        constructor_params_encoded = b'' + eth_abi.encode_single('int', 1)
        deployed_contract_addr = simulate_tx("3,miner,{}".format(contract_bytecode + constructor_params_encoded.hex()), w3)
        token_contracts[deployed_contract_addr] = w3.eth.contract(abi=temp_abi, address=deployed_contract_addr)

        # Preparation transactions
        for token in approved_tokens:
            if 'sushiswap' in involved_dexes:
                approve_tx = '1,miner,{},0,approve,{},1000000000000000000000000000'.format(token, sushiswap_router_contract.address)
                simulate_tx(approve_tx, w3) #1e27
            if 'uniswapv2' in involved_dexes:
                approve_tx = '1,miner,{},0,approve,{},1000000000000000000000000000'.format(token, uniswap_router_contract.address)
                simulate_tx(approve_tx, w3) #1e27
            if 'uniswapv3' in involved_dexes:
                approve_tx = '1,miner,{},0,approve,{},1000000000000000000000000000'.format(token, uniswapv3_router_contract.address)
                simulate_tx(approve_tx, w3) #1e27
                approve_tx = '1,miner,{},0,approve,{},1000000000000000000000000000'.format(WETH, uniswapv3_router_contract.address)
                simulate_tx(approve_tx, w3) #1e27
                approve_tx = '1,miner,{},{},deposit'.format(WETH, int(MINER_CAPITAL/2/1e18))
                simulate_tx(approve_tx, w3) #1e27
   
    # Execute transactions
    for line in lines[MY_SETUP_LINES:]:
        if line.startswith('#'):
            continue
        simulate_tx(line, w3)
  
    simulate_tx("1,miner," + deployed_contract_addr + ",1,set_age,100", w3)
    simulate_tx("1,miner," + deployed_contract_addr + ",1,set_age,99", w3)
 
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
            mev = max(mev, BLOCKREWARD + get_mev_dex(token_addr, remaining_balance, w3, involved_dexes))  # blockreward for parity with dex
            # uniswapv2_out_amount = getAmountOutv2(token_addr, uniswap_router_contract, remaining_balance)
            # sushiswap_out_amount = getAmountOutv2(token_addr, sushiswap_router_contract, remaining_balance)
            # if sushiswap_out_amount > uniswapv2_out_amount:
            #     automatic_tx = '1,miner,SushiswapRouter,0,swapExactTokensForETH,{},0,[{}-{}],miner,1800000000'.format(remaining_balance, token_addr, WETH)
            # else:
            #     automatic_tx = '1,miner,UniswapV2Router,0,swapExactTokensForETH,{},0,[{}-{}],miner,1800000000'.format(remaining_balance, token_addr, WETH)
            #TODO : dont hardcode fees, check which fee pool is the most profitable
            # fee_pool = get_best_fee_pool(token_addr, remaining_balance, w3)
            # automatic_tx = '1,miner,UniswapV3Router,0,exactInputSingle,{},{},{},miner,1800000000,{},0,0'.format(token_addr, WETH, fee_pool, remaining_balance)
            # simulate_tx(automatic_tx, w3)
        # mine the block to execute automatic_tx
        # mine_block()
        # mev = max(mev, get_mev())

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
    setup(lines, port_id, ['uniswapv2', 'uniswapv3'])
    print("simulating...")



    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()
    for i in range(100):
        print("Round #", i);
        total_time = 0
        current = time.time()
        mev = simulate(lines, port_id, ['uniswapv2', 'uniswapv3'], False, '', args.settlement)
        print(mev)
        print("single run time: ", time.time() - current)
        print("time for rpc calls: ", total_time);

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
