#!/bin/sh

cd ../

# kill -s 9 `ps -ef | grep src.flora.test.launch_training |grep -v grep | awk '{print $2}'`
# kill -9 $(ps aux | grep src.flora.test.launch_training | grep -v grep | awk '{print $2}')

#dir='/Users/ssq/Desktop/datasets/flora_test/'
dir='/ccsopen/home/ssq/datasets/'
interface='eth1'
bsz=32
testbsz=32
worldsize=8
#comm='RPC'
comm='Collective'
algo='fedavg'
commfreq=500
masteraddr='127.0.0.1'
masterport=28670
#masterport=50055
backend='gloo'
lr=0.01
gamma=0.2
momentum=0.9
weightdecay=5e-4
model='vgg11'
dataset='cifar100'

for val in $(seq 1 $worldsize)
do
  rank=$(($val-1))
  echo '###### going to launch training for rank '$rank
  python3 -m src.flora.test.launch_training --dir=$dir --bsz=$bsz --rank=$rank --world-size=$worldsize \
  --communicator=$comm --comm-freq=$commfreq --master-addr=$masteraddr --master-port=$masterport --backend=$backend \
  --model=$model --dataset=$dataset --train-dir=$dir --test-dir=$dir --algo=$algo --lr=$lr --gamma=$gamma \
  --weight-decay=$weightdecay --momentum=$momentum --test-bsz=$testbsz --network-interface=$interface &
  echo "going to sleep for 2 seconds..."
  sleep 3
done