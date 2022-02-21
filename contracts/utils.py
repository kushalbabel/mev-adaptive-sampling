from web3 import Web3
import json
import eth_abi

def encode_parameters(contract, fn_name, params):
    fn = contract.find_functions_by_name(fn_name)[0]
    return eth_abi.encode_abi([arg['type'] for arg in fn.abi['inputs']], params)

def encode_function_signature(contract, fn_name):
    fn = contract.find_functions_by_name(fn_name)[0]
    types = [arg['type'] for arg in fn.abi['inputs']]
    selector =  Web3.sha3(text='{}({})'.format(fn_name,','.join(types)))[0:4]
    return selector

def encode_function_call(contract, fn_name, params):
    return encode_function_signature(contract, fn_name).hex() + encode_parameters(contract, fn_name, params).hex()