from web3 import Web3
import json
import eth_abi
from pathlib import Path

w3 = Web3()
path = Path(__file__).parent / "erc20_abi.json"
erc20_abi = json.loads(path.open('r').read()) #TODO: weird abs path, make pretty

path = Path(__file__).parent / "weth_abi.json"
weth_abi = json.loads(path.open('r').read()) #TODO: weird abs path, make pretty

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

