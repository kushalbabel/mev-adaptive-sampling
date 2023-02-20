import pandas as pd

df = pd.read_csv('/data/latest-data/uniswapv2_pairs.csv')

eth = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
uniswapv2 = "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f"
sushiswap = "0xc0aee478e3658e2610c5f7a4a2e1777ce9e4f2ac"

def uniswap_to_sushiswap(uniswap_addr):
    row = df[df["pair"] == uniswap_addr].iloc[0]
    a = row.token0
    b = row.token1
    target_row = df[(df["exchange"] == sushiswap) & (df["token0"] == a) & (df["token1"] == b) ].iloc[0]
    return target_row.pair

def sushiswap_to_uniswap(sushiswap_addr):
    row = df[df["pair"] == sushiswap_addr].iloc[0]
    a = row.token0
    b = row.token1
    target_row = df[(df["exchange"] == uniswapv2) & (df["token0"] == a) & (df["token1"] == b) ].iloc[0]
    return target_row.pair

def sushiswap_to_uniswapv3(sushiswap_addr):
    row = df[df["pair"] == sushiswap_addr].iloc[0]
    a = row.token0
    b = row.token1
    if a == eth:
        return b
    elif b == eth:
        return a

def uniswapv3_to_sushiswap(uniswapv3_addr):
    target_row = df[(df["exchange"] == sushiswap) & ((df["token0"] == eth) & (df["token1"] == uniswapv3_addr) 
                    | (df["token0"] == uniswapv3_addr) & (df["token1"] == eth))].iloc[0]
    return target_row.pair

# print(uniswapv3_to_sushiswap('0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48'))
# print(sushiswap_to_uniswapv3('0x397ff1542f962076d0bfe58ea045ffa2d347aca0'))