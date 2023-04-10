#44 Clients run on port 8547 - 8590

##start all clients:
`bash launch_hardhats.sh`


##Check if clients are running:
`bash ping_hardhats.sh` -> Ping clients by issuing a request for getting the current block number

If the client is running, it will return a json response with the "result"
If the client is stuck, the command will get stuck.
If the client is not running, output will say "refused to connect"


##kill all clients:
`bash kill_hardhat.sh` -> Stop all clients

##Restart only one client:
`bash restart_hardhat.sh <PORT>`
eg. `bash restart_hardhat.sh 8547`

## Check logs of clients:
eg. `tail -f logs/hardhat.8547`

## Check all clients are being used via the logging time stamps:
ls -al /home/kb742/mev-adaptive-sampling/eth_clients/hardhat/logs/


0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc_14054805   -> gives 44/44 Nan scores


# Speed up harthat simulation

## Set up ethereumjs-evm and ethereumjs-vm
```
cd ~/
git clone https://github.com/iseriohn/ethereumjs-monorepo
cd ethereumjs-monorepo
git checkout ugly
./setup.sh
```

## Set up hardhat
```
cd ~/
git clone https://github.com/iseriohn/hardhat
cd hardhat
git checkout simtx
./setup.sh

cd packages/hardhat-core && yarn link && yarn build
```


## Init a new node project
```
cd ~/
mkdir tmp
cd tmp
npm init
cp ~/mev-adaptive-sampling/eth_clients/hardhat/hardhat.config.js ./


yarn link hardhat
yarn add hardhat
npx hardhat node --port 8600 --verbose
```

## Run the simulation script
```
python simulate_client.py -f manualtests/optimised_2 -p 53
```