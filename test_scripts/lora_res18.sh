#!/bin/sh

cd ../

# kill -s 9 `ps -ef | grep src.flora.test.launch_lora |grep -v grep | awk '{print $2}'`
# kill -9 $(ps aux | grep src.flora.test.launch_lora | grep -v grep | awk '{print $2}')

#dir='/Users/ssq/Desktop/datasets/flora_test/'
#worldsize=2
#interface='lo0'
#masterport=28670

dir='/ccsopen/home/ssq/datasets/'
masterport=28360
worldsize=8
interface='eth1'

bsz=32
masteraddr='127.0.0.1'
backend='gloo'
model='resnet18'
dataset='cifar10'
compression='PowerSGD'
poweritr=5
#compressrank=64
#mincompressrate=20
compressrank=16
mincompressrate=10

for val in $(seq 1 $worldsize)
do
  rank=$(($val-1))
  echo '###### going to launch training for rank '$rank
  python3 -m src.flora.test.launch_lora --dir=$dir --bsz=$bsz --rank=$rank --world-size=$worldsize \
  --master-addr=$masteraddr --master-port=$masterport --backend=$backend --model=$model --dataset=$dataset \
  --train-dir=$dir --test-dir=$dir --compression-type=$compression --network-interface=$interface \
  --power-itr=$poweritr --compress-rank=$compressrank --min-compression-rate=$mincompressrate &
  sleep 2
done