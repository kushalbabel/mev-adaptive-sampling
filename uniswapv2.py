import re
from collections import defaultdict 
from math import sqrt

class UniswapV2:
    def __init__(self, balances={}, exchange_name='UniswapV2'):
        self.exchange_name = exchange_name
        self.token_balances = defaultdict(lambda : defaultdict(lambda : 0))
        self.token_balances[self.exchange_name] = balances
        self.lp_token = 'lp_token'
        self.supply = 0
        self.min_liquidity = 0.00001 #depends on the decimals, rn deminals = 1 instead of 18 or 6

    def process(self, tx):
        tx = tx.replace(';', '').strip()
        
        if 'adds' in tx:
            self.add_liquidity(tx)
        elif 'removes' in tx:
            self.remove_liquidity(tx)
        elif 'swaps' in tx:
            self.swap(tx)
        elif 'redeems' in tx:
            self.redeem(tx)
        elif tx.startswith('//'):
            pass
        else:
            print("ILLEGAL ", tx)

    def add_liquidity(self, tx):
        vals = re.match(r'(.*) adds (.*) (.*) and (.*) (.*) of liquidity', tx)
        token0 = vals.group(3)
        token1 = vals.group(5)
        amount0 = float(vals.group(2))
        amount1 = float(vals.group(4))
        address = vals.group(1)
        if (self.token_balances[self.exchange_name][token0] == 0) :
            lp_tokens = sqrt(amount0*amount1)
            self.token_balances[address][self.lp_token] += lp_tokens - self.min_liquidity
            self.supply += lp_tokens
        else:
            lp_tokens = min(amount0*self.supply/self.token_balances[self.exchange_name][token0], amount1*self.supply/self.token_balances[self.exchange_name][token1])
            self.supply += lp_tokens
            self.token_balances[address][self.lp_token] += lp_tokens
        self.token_balances[self.exchange_name][token0] += amount0
        self.token_balances[self.exchange_name][token1] += amount1
        self.token_balances[address][token0] -= amount0
        self.token_balances[address][token1] -= amount1
        
    def remove_liquidity(self, tx):
        vals = re.match(r'(.*) removes (.*) (.*) and (.*) (.*) of liquidity', tx)
        token0 = vals.group(3)
        token1 = vals.group(5)
        amount0 = float(vals.group(2))
        amount1 = float(vals.group(4))
        address = vals.group(1)
        # in this model we dont revert when removing more than deposited, use redeem if need to revert
        self.token_balances[self.exchange_name][token0] -= amount0
        self.token_balances[self.exchange_name][token1] -= amount1
        self.token_balances[address][token0] += amount0
        self.token_balances[address][token1] += amount1

    def redeem(self, tx):
        vals = re.match(r'(.*) redeems (.*) fraction of liquidity from (.*) and (.*)', tx)
        lp_fraction = float(vals.group(2))
        address = vals.group(1)
        token0 = vals.group(3)
        token1 = vals.group(4)
        held_lp_tokens = self.token_balances[address][self.lp_token]
        redeemed_lp_tokens = lp_fraction*held_lp_tokens / 100.0
        amount0 = redeemed_lp_tokens*self.token_balances[self.exchange_name][token0]/self.supply
        amount1 = redeemed_lp_tokens*self.token_balances[self.exchange_name][token1]/self.supply
        self.token_balances[self.exchange_name][token0] -= amount0
        self.token_balances[self.exchange_name][token1] -= amount1
        self.token_balances[address][token0] += amount0
        self.token_balances[address][token1] += amount1
        self.supply -= redeemed_lp_tokens
        self.token_balances[address][self.lp_token] -= redeemed_lp_tokens

        
    def swap(self, tx):
        vals = re.match(r'(.*) swaps for (.*) by providing (.*) (.*) and (.*) (.*) with change (.*) fee (.*)', tx)
        address = vals.group(1)
        token_in = vals.group(4)
        token_out = vals.group(6)
        amount_in_token_in = float(vals.group(3))
        amount_in_token_out = float(vals.group(5))
        amount_out_token_in = float(vals.group(7))

        amount_out_token_out = (((997 * amount_in_token_in - 1000 * amount_out_token_in) * self.token_balances[self.exchange_name][token_out]) // (1000 * (self.token_balances[self.exchange_name][token_in] - amount_out_token_in) + 997 * amount_in_token_in)) + ((amount_in_token_out * 997) // (1000))
        
        self.token_balances[self.exchange_name][token_in] += amount_in_token_in - amount_out_token_in
        self.token_balances[self.exchange_name][token_out] += amount_in_token_out - amount_out_token_out
        self.token_balances[address][token_in] += amount_out_token_in - amount_in_token_in
        self.token_balances[address][token_out] += amount_out_token_out - amount_in_token_out

    
    def config(self):
        return self.token_balances

    def reserves(self):
        return self.token_balances[self.exchange_name]
