import requests
import glob
import json

filename = 'flashbots_data_on_demand.json'
existing_data = set()
f = open(filename, 'r')
data = json.load(f)
f.close()

# print(data)

start_block = 14986955

for artifact in glob.glob('/home/gid-javaheripim/mev-adaptive-sampling/artifacts_smooth/*/*/'):
    block_number = artifact.split('/')[-2]
    if block_number not in data and int(block_number) > start_block:
        print(block_number)
        r = requests.get('https://blocks.flashbots.net/v1/blocks?block_number={}'.format(block_number))
        try:
            blocks = r.json()['blocks']
            if len(blocks) == 0:
                data[block_number] = {}
            else:    
                data[block_number] = blocks[0]
        except:
            print(r.content)
            break

json.dump(data, open(filename, 'w'))