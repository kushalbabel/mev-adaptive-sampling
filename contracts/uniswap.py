from web3 import Web3
import json
import eth_abi

w3 = Web3()
uniswap_router_abi = json.loads(open('contracts/uniswap_router_abi.json','r').read()) #TODO: weird abs path, make pretty
uniswap_router_contract = w3.eth.contract(abi=uniswap_router_abi, address='0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D')
sushiswap_router_contract = w3.eth.contract(abi=uniswap_router_abi, address='0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F')