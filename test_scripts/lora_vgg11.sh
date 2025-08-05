#!/bin/sh

cd ../

# kill -s 9 `ps -ef | grep src.flora.test.launch_lora |grep -v grep | awk '{print $2}'`
# kill -9 $(ps aux | grep src.flora.test.launch_lora | grep -v grep | awk '{print $2}')

#dir='/Users/ssq/Desktop/datasets/flora_test/'
#worldsize=2
#interface='lo0'
#masterport=28670

dir='/ccsopen/home/ssq/datasets/'
masterport=28640
worldsize=4
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
compression='PowerSGD'
poweritr=5

#compressrank=64
#mincompressrate=20

compressrank=32
mincompressrate=20

for val in $(seq 1 $worldsize)
do
  rank=$(($val-1))
  echo '###### going to launch training for rank '$rank
  python3 -m src.flora.test.launch_lora --dir=$dir --bsz=$bsz --rank=$rank --world-size=$worldsize \
  --master-addr=$masteraddr --master-port=$masterport --backend=$backend --model=$model --dataset=$dataset \
  --train-dir=$dir --test-dir=$dir --compression-type=$compression --lr=$lr --gamma=$gamma \
  --weight-decay=$weightdecay --momentum=$momentum --test-bsz=$testbsz --network-interface=$interface \
  --power-itr=$poweritr --compress-rank=$compressrank --min-compression-rate=$mincompressrate &
  sleep 3
done