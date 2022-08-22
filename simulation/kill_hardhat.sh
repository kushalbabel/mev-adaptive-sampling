#!/bin/bash
CORES=44
for ((i=0;i<CORES;i++)); do
	PORT=$(( 8547 + $i ))
	echo $PORT
	pid=`ps -ef | grep hardhat | grep $PORT | awk {'print $2'}`
	echo $pid
	kill -s SIGTERM $pid
done
