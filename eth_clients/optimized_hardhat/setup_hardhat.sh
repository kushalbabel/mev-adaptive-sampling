#!/bin/bash
CORES=40
for ((i=0;i<CORES;i++)); do
    PORT=$(( 8601 + $i ))
    dir=$PORT
    mkdir -p $dir
    cd $dir
    rm -rf cache/
    rm -rf node_modules/
    cp ../hardhat.config.js ./
    cp ../yarn.lock ./
    cd ..
	# eval $cmd
done
