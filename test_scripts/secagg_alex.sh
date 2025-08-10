#!/bin/sh

cd ../

# kill -s 9 `ps -ef | grep src.flora.privacy.launch_secagg |grep -v grep | awk '{print $2}'`
# kill -9 $(ps aux | grep src.flora.privacy.launch_secagg | grep -v grep | awk '{print $2}')

#dir='/Users/ssq/Desktop/datasets/flora_test/'
#interface='lo0'
#worldsize=2
#masterport=28670

dir='/ccsopen/home/ssq/datasets/'
masterport=28670
interface='eth1'
worldsize=2

bsz=32
testbsz=128
comm='Collective'
commfreq=100
masteraddr='127.0.0.1'
backend='gloo'
model='alexnet'
dataset='caltech101'
lr=0.01
gamma=0.1
weightdecay=5e-4
momentum=0.9

for val in $(seq 1 $worldsize)
do
  rank=$(($val-1))
  echo '###### going to launch training for rank '$rank
  python3 -m src.flora.privacy.launch_secagg --dir=$dir --bsz=$bsz --rank=$rank --world-size=$worldsize \
  --communicator=$comm --comm-freq=$commfreq --master-addr=$masteraddr --master-port=$masterport --backend=$backend \
  --model=$model --dataset=$dataset --train-dir=$dir --test-dir=$dir --lr=$lr --gamma=$gamma \
  --weight-decay=$weightdecay --momentum=$momentum --test-bsz=$testbsz --network-interface=$interface &
  echo "going to sleep for 2 seconds..."
  sleep 2
done