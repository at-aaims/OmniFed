#!/bin/sh

cd ../

#


# kill -9 $(ps aux | grep src.flora.test.launch_training | grep -v grep | awk '{print $2}')

#dir='/Users/ssq/Desktop/datasets/flora_test/'
dir='/ccsopen/home/ssq/datasets/'
interface='eth1'
bsz=32
worldsize=8
#comm='RPC'
comm='Collective'
algo='fednova'
commfreq=500
masteraddr='127.0.0.1'
masterport=28670
#masterport=50055
backend='gloo'
model='resnet18'
dataset='cifar10'

#python3 -m src.flora.test.test1
for val in $(seq 1 $worldsize)
do
  rank=$(($val-1))
  echo '###### going to launch training for rank '$rank
  python3 -m src.flora.test.launch_training --dir=$dir --bsz=$bsz --rank=$rank --world-size=$worldsize \
  --communicator=$comm --comm-freq=$commfreq --master-addr=$masteraddr --master-port=$masterport --backend=$backend \
  --model=$model --dataset=$dataset --train-dir=$dir --test-dir=$dir --algo=$algo --network-interface=$interface &
  echo "going to sleep for 2 seconds..."
  sleep 2
done