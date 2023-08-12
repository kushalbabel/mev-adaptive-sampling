def swap_template(template_variable, token0, token1):
    return '1,miner,SushiswapRouter,{},swapExactETHForTokens,0,[{}-{}],miner,1800000000'.format(template_variable, token0, token1)
