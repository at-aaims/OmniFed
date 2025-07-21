#!/bin/sh

cd ../

# kill -s 9 `ps -ef | grep src.flora.test.launch_sparsification |grep -v grep | awk '{print $2}'`
# kill -9 $(ps aux | grep src.flora.test.launch_sparsification | grep -v grep | awk '{print $2}')

dir='/ccsopen/home/ssq/datasets/'
interface='eth1'
worldsize=8
#dir='/Users/ssq/Desktop/datasets/flora_test/'
#interface='lo0'
#worldsize=4
bsz=32
masteraddr='127.0.0.1'
#masterport=28670
masterport=29860
backend='Gloo'
model='resnet18'
dataset='cifar10'
compression='topK'
compressratio=0.1

for val in $(seq 1 $worldsize)
do
  rank=$(($val-1))
  echo '###### going to launch training for rank '$rank
  python3 -m src.flora.test.launch_sparsification --dir=$dir --bsz=$bsz --rank=$rank --world-size=$worldsize \
  --master-addr=$masteraddr --master-port=$masterport --backend=$backend --model=$model --dataset=$dataset \
  --train-dir=$dir --test-dir=$dir --compression-type=$compression --compress-ratio=$compressratio \
  --network-interface=$interface &
  sleep 3
done