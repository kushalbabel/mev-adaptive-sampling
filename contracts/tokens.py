from web3 import Web3
import json
import eth_abi

w3 = Web3()
erc20_abi = json.loads(open('contracts/erc20_abi.json','r').read()) #TODO: weird abs path, make pretty


usdc_contract = w3.eth.contract(abi=erc20_abi, address='0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48')
usdt_contract = w3.eth.contract(abi=erc20_abi, address='0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48')
wbtc_contract = w3.eth.contract(abi=erc20_abi, address='0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48')
link_contract = w3.eth.contract(abi=erc20_abi, address='0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48')

token_contracts = {
    'usdc': usdc_contract,
    'usdt': usdt_contract,
    'link': link_contract,
    'wbtc': wbtc_contract
}

