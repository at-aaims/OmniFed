#!/bin/sh

cd ../

# kill -s 9 `ps -ef | grep src.flora.test.launch_sparsification |grep -v grep | awk '{print $2}'`
# kill -9 $(ps aux | grep src.flora.test.launch_sparsification | grep -v grep | awk '{print $2}')

#dir='/Users/ssq/Desktop/datasets/flora_test/'
#masterport=28670
#worldsize=3
#interface='lo0'

#dir='/ccsopen/home/ssq/datasets/'
#masterport=29110
dir='/ccsopen/home/ssq/datasets2/'
masterport=29118
#dir='/ccsopen/home/ssq/datasets3/'
#masterport=27340
#dir='/ccsopen/home/ssq/datasets4/'
#masterport=26290
interface='eth1'
worldsize=8
bsz=32
testbsz=128
masteraddr='127.0.0.1'
backend='gloo'
model='alexnet'
dataset='caltech101'
lr=0.01
gamma=0.1
weightdecay=5e-4
momentum=0.9
compression='topK'
compressratio=0.1

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