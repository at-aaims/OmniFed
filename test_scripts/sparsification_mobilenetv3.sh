#!/bin/sh

cd ../

# kill -s 9 `ps -ef | grep src.flora.test.launch_sparsification |grep -v grep | awk '{print $2}'`
# kill -9 $(ps aux | grep src.flora.test.launch_sparsification | grep -v grep | awk '{print $2}')

#dir='/Users/ssq/Desktop/datasets/flora_test/'
#worldsize=4
#interface='lo0'
#masterport=28670

#dir='/ccsopen/home/ssq/datasets/'
#masterport=24870
#dir='/ccsopen/home/ssq/datasets2/'
#masterport=29860
dir='/ccsopen/home/ssq/datasets3/'
masterport=22180
#dir='/ccsopen/home/ssq/datasets4/'
#masterport=26290
worldsize=8
interface='eth1'
bsz=32
testbsz=32
masteraddr='127.0.0.1'
backend='gloo'
model='mobilenetv3'
dataset='caltech256'
lr=0.1
gamma=0.1
weightdecay=1e-4
momentum=0.9
lrstepsize=40
numclasses=257
compression='topK'
compressratio=0.1

for val in $(seq 1 $worldsize)
do
  rank=$(($val-1))
  echo '###### going to launch training for rank '$rank
  python3 -m src.flora.test.launch_sparsification --dir=$dir --bsz=$bsz --rank=$rank --world-size=$worldsize \
  --master-addr=$masteraddr --master-port=$masterport --backend=$backend --model=$model --dataset=$dataset \
  --train-dir=$dir --test-dir=$dir --compression-type=$compression --compress-ratio=$compressratio --lr=$lr \
  --gamma=$gamma --weight-decay=$weightdecay --momentum=$momentum --test-bsz=$testbsz \
  --mobv3-lr-step-size=$lrstepsize --mobv3-num-classes=$numclasses --network-interface=$interface &
  sleep 3
done