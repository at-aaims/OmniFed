#!/bin/sh

cd ../

# kill -s 9 `ps -ef | grep src.flora.privacy.launch_dp |grep -v grep | awk '{print $2}'`
# kill -9 $(ps aux | grep src.flora.privacy.launch_dp | grep -v grep | awk '{print $2}')

#dir='/Users/ssq/Desktop/datasets/flora_test/'
#commfreq=10
#interface='lo0'
#masterport=28670
#worldsize=4

dir='/ccsopen/home/ssq/datasets/'
interface='eth1'
masterport=28670
worldsize=8

bsz=32
testbsz=32
comm='Collective'
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
epochs=80

#epsilon=1.0
#delta=1e-5
#gamma=0.01

epsilon=25.0
delta=2e-5
gamma=0.1

for val in $(seq 1 $worldsize)
do
  rank=$(($val-1))
  echo '###### going to launch training for rank '$rank
  python3 -m src.flora.privacy.launch_dp --dir=$dir --bsz=$bsz --rank=$rank --world-size=$worldsize \
  --communicator=$comm --master-addr=$masteraddr --master-port=$masterport --backend=$backend \
  --model=$model --dataset=$dataset --train-dir=$dir --test-dir=$dir --lr=$lr --gamma=$gamma \
  --weight-decay=$weightdecay --momentum=$momentum --test-bsz=$testbsz --mobv3-lr-step-size=$lrstepsize \
  --mobv3-num-classes=$numclasses --network-interface=$interface --dp-epsilon=$epsilon --dp-delta=$delta \
  --dp-gamma=$gamma --epochs=$epochs &
  echo "going to sleep for 2 seconds..."
  sleep 2
done