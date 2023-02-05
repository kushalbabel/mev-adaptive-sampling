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
