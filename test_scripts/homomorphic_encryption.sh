#!/bin/sh

cd ../

# kill -s 9 `ps -ef | grep src.flora.test.launch_he |grep -v grep | awk '{print $2}'`
# kill -9 $(ps aux | grep src.flora.test.launch_he | grep -v grep | awk '{print $2}')

dir='/Users/ssq/Desktop/datasets/flora_test/'
#dir='/ccsopen/home/ssq/datasets/'
#interface='eth1'
bsz=32
worldsize=1
comm='Collective'
masteraddr='127.0.0.1'
masterport=28670
backend='gloo'
model='resnet18'
dataset='cifar10'

for val in $(seq 1 $worldsize)
do
  rank=$(($val-1))
  echo '###### going to launch homomorphic encryption on rank '$rank
  python3 -m src.flora.privacy.launch_he --dir=$dir --bsz=$bsz --rank=$rank --world-size=$worldsize \
  --communicator=$comm --master-addr=$masteraddr --master-port=$masterport --backend=$backend \
  --model=$model --dataset=$dataset --train-dir=$dir --test-dir=$dir &
  echo "going to sleep for 2 seconds..."
  sleep 2
done