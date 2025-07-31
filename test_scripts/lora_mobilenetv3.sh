#!/bin/sh

cd ../

# kill -s 9 `ps -ef | grep src.flora.test.launch_lora |grep -v grep | awk '{print $2}'`
# kill -9 $(ps aux | grep src.flora.test.launch_lora | grep -v grep | awk '{print $2}')

#dir='/Users/ssq/Desktop/datasets/flora_test/'
#masterport=28670
#interface='lo0'
#worldsize=4

dir='/ccsopen/home/ssq/datasets/'
masterport=25783
interface='eth1'
worldsize=8

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
  --weight-decay=$weightdecay --momentum=$momentum --test-bsz=$testbsz --mobv3-lr-step-size=$lrstepsize \
  --mobv3-num-classes=$numclasses --network-interface=$interface --power-itr=$poweritr \
  --compress-rank=$compressrank --min-compression-rate=$mincompressrate &
  sleep 3
done