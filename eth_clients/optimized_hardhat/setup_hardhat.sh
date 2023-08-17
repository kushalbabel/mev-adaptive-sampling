#!/bin/bash
CORES=40
for ((i=0;i<CORES;i++)); do
    PORT=$(( 8601 + $i ))
    dir=$PORT
    mkdir -p $dir
    cd $dir
    npm init -y
    cp ../hardhat.config.js ./
    yarn add hardhat && rm -rf node_modules/ yarn.lock && yarn link hardhat && yarn add hardhat
    cd ..
	# eval $cmd
done
