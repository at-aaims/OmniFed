#!/bin/sh

cd ../

# kill -s 9 `ps -ef | grep src.flora.test.launch_quantization |grep -v grep | awk '{print $2}'`
# kill -9 $(ps aux | grep src.flora.test.launch_quantization | grep -v grep | awk '{print $2}')

#dir='/Users/ssq/Desktop/datasets/flora_test/'
#worldsize=3
#interface='lo0'
#masterport=28670

dir='/ccsopen/home/ssq/datasets/'
masterport=25783
#dir='/ccsopen/home/ssq/datasets2/'
#masterport=29189
#dir='/ccsopen/home/ssq/datasets3/'
#masterport=28139
#dir='/ccsopen/home/ssq/datasets4/'
#masterport=26290
worldsize=8
interface='eth1'
bsz=32
masteraddr='127.0.0.1'
backend='gloo'
model='resnet18'
dataset='cifar10'
compression='QSGD'
bitwidth=8

for val in $(seq 1 $worldsize)
do
  rank=$(($val-1))
  echo '###### going to launch training for rank '$rank
  python3 -m src.flora.test.launch_quantization --dir=$dir --bsz=$bsz --rank=$rank --world-size=$worldsize \
  --master-addr=$masteraddr --master-port=$masterport --backend=$backend --model=$model --dataset=$dataset \
  --train-dir=$dir --test-dir=$dir --compression-type=$compression --quantized-bitwidth=$bitwidth \
  --network-interface=$interface &
  sleep 3
done