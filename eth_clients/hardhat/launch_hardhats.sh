#!/bin/bash
CORES=40
for ((i=0;i<CORES;i++)); do
	PORT=$(( 8547 + $i ))
	cmd="nohup npx hardhat node --port $PORT &> logs/hardhat.$PORT &"
	echo $cmd
	eval $cmd
done
