#!/bin/sh

cd ../

# kill -s 9 `ps -ef | grep src.flora.test.launch_training |grep -v grep | awk '{print $2}'`
# kill -9 $(ps aux | grep src.flora.test.launch_training | grep -v grep | awk '{print $2}')

#dir='/Users/ssq/Desktop/datasets/flora_test/'
#commfreq=10
#interface='lo0'
#worldsize=4
dir='/ccsopen/home/ssq/datasets/'
interface='eth1'
commfreq=100
worldsize=4
bsz=32
testbsz=32
#comm='RPC'
comm='Collective'
algo='fedmom'
masteraddr='127.0.0.1'
masterport=28670
#masterport=50055
backend='gloo'
model='mobilenetv3'
dataset='caltech256'
lr=0.1
gamma=0.1
weightdecay=1e-4
momentum=0.9
lrstepsize=40
numclasses=257

for val in $(seq 1 $worldsize)
do
  rank=$(($val-1))
  echo '###### going to launch training for rank '$rank
  python3 -m src.flora.test.launch_training --dir=$dir --bsz=$bsz --rank=$rank --world-size=$worldsize \
  --communicator=$comm --comm-freq=$commfreq --master-addr=$masteraddr --master-port=$masterport --backend=$backend \
  --model=$model --dataset=$dataset --train-dir=$dir --test-dir=$dir --algo=$algo --lr=$lr --gamma=$gamma \
  --weight-decay=$weightdecay --momentum=$momentum --test-bsz=$testbsz --mobv3-lr-step-size=$lrstepsize \
  --mobv3-num-classes=$numclasses --network-interface=$interface &
  echo "going to sleep for 3 seconds..."
  sleep 3
done