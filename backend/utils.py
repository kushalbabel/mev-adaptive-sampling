import csv
import subprocess

data_path = '/home/kb742/mev-adaptive-sampling/data'
binance_prices_path = f'{data_path}/prices-binance/'
block_times = f'{data_path}/block_times.csv'
token_names = f'{data_path}/token_names.csv'

def get_price(block, token_addr,source='binance'):
    # TODO: find the token price in dollar terms
    if source=='binance':
        if token_addr == 'eth':
            market = 'ETHUSDT'
        else:
            matched = subprocess.check_output(f'grep {token_addr.lower()} {token_names}', shell=True)
            token_name = str(matched, "utf-8").strip().split(",")[1]
            market = "f{token_name}USDT"
        #TODO check if token name market exists
        matched_ts = subprocess.check_output(f'grep {block} {block_times}', shell=True)
        ts = int(str(matched_ts, "utf-8").strip().split(",")[1])
        minute_ts = ts//60 * 60
        matched_price = subprocess.check_output(f'grep {minute_ts*1000} {binance_prices_path}{market}.csv', shell=True)
        price = float(str(matched_price, "utf-8").strip().split(",")[1])
        return price