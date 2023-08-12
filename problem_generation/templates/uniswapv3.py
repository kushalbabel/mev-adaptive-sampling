def swap_templates(template_variables, token0, token1):
    assert(len(template_variables) == 3)
    swap_template1 = '1,miner,UniswapV3Router,0,exactInputSingle,{},{},{},miner,1800000000,{},0,0'.format(token0, token1, 500, template_variables[0])
    swap_template2 = '1,miner,UniswapV3Router,0,exactInputSingle,{},{},{},miner,1800000000,{},0,0'.format(token0, token1, 3000, template_variables[1])
    swap_template3 = '1,miner,UniswapV3Router,0,exactInputSingle,{},{},{},miner,1800000000,{},0,0'.format(token0, token1, 10000, template_variables[2])
    return [swap_template1, swap_template2, swap_template3]