#!/bin/sh

cd ../

# kill -s 9 `ps -ef | grep src.flora.test.launch_sparsification |grep -v grep | awk '{print $2}'`
# kill -9 $(ps aux | grep src.flora.test.launch_sparsification | grep -v grep | awk '{print $2}')

#dir='/Users/ssq/Desktop/datasets/flora_test/'
#masterport=28670
#worldsize=3
interface='lo0'
dir='/ccsopen/home/ssq/datasets2/'
masterport=28670
worldsize=8
interface='eth1'
bsz=32
testbsz=32
masteraddr='127.0.0.1'
backend='gloo'
lr=0.01
gamma=0.2
momentum=0.9
weightdecay=5e-4
model='vgg11'
dataset='cifar100'
compression='topK'
compressratio=0.001

for val in $(seq 1 $worldsize)
do
  rank=$(($val-1))
  echo '###### going to launch training for rank '$rank
  python3 -m src.flora.test.launch_sparsification --dir=$dir --bsz=$bsz --rank=$rank --world-size=$worldsize \
  --master-addr=$masteraddr --master-port=$masterport --backend=$backend --model=$model --dataset=$dataset \
  --train-dir=$dir --test-dir=$dir --compression-type=$compression --compress-ratio=$compressratio --lr=$lr \
  --gamma=$gamma --weight-decay=$weightdecay --momentum=$momentum --test-bsz=$testbsz --network-interface=$interface &
  sleep 3
done