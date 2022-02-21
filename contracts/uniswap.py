from web3 import Web3
import json
import eth_abi

w3 = Web3()
uniswap_router_abi = json.loads(open('uniswap_router_abi.json','r').read())
uniswap_router_contract = w3.eth.contract(abi=uniswap_router_abi)