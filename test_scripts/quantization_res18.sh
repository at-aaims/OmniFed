#!/bin/sh

cd ../

# kill -s 9 `ps -ef | grep src.flora.test.launch_quantization |grep -v grep | awk '{print $2}'`
# kill -9 $(ps aux | grep src.flora.test.launch_quantization | grep -v grep | awk '{print $2}')

dir='/Users/ssq/Desktop/datasets/flora_test/'
bsz=32
worldsize=3
masteraddr='127.0.0.1'
masterport=28670
backend='Gloo'
model='resnet18'
dataset='cifar10'
compression='AMP'
bitwidth=8

for val in $(seq 1 $worldsize)
do
  rank=$(($val-1))
  echo '###### going to launch training for rank '$rank
  python3 -m src.flora.test.launch_quantization --dir=$dir --bsz=$bsz --rank=$rank --world-size=$worldsize \
  --master-addr=$masteraddr --master-port=$masterport --backend=$backend --model=$model --dataset=$dataset \
  --train-dir=$dir --test-dir=$dir --compression-type=$compression --quantized-bitwidth=$bitwidth &
  sleep 3
done