#!/bin/sh

cd ../

# kill -s 9 `ps -ef | grep src.flora.test.launch_compression |grep -v grep | awk '{print $2}'`
# kill -9 $(ps aux | grep src.flora.test.launch_compression | grep -v grep | awk '{print $2}')

dir='/Users/ssq/Desktop/datasets/flora_test/'
bsz=32
testbsz=32
worldsize=3
masteraddr='127.0.0.1'
masterport=28670
backend='Gloo'
lr=0.01
gamma=0.2
momentum=0.9
weightdecay=5e-4
model='vgg11'
dataset='cifar100'
compression='randomK'
compressratio=0.1

for val in $(seq 1 $worldsize)
do
  rank=$(($val-1))
  echo '###### going to launch training for rank '$rank
  python3 -m src.flora.test.launch_compression --dir=$dir --bsz=$bsz --rank=$rank --world-size=$worldsize \
  --master-addr=$masteraddr --master-port=$masterport --backend=$backend --model=$model --dataset=$dataset \
  --train-dir=$dir --test-dir=$dir --compression-type=$compression --compress-ratio=$compressratio --lr=$lr \
  --gamma=$gamma --weight-decay=$weightdecay --momentum=$momentum --test-bsz=$testbsz &
  sleep 3
done