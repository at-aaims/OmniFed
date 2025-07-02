#!/bin/sh

# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cd ../

# kill -s 9 `ps -ef | grep src.flora.test.launch_hybridcomm |grep -v grep | awk '{print $2}'`
# kill -9 $(ps aux | grep src.flora.test.launch_hybridcomm | grep -v grep | awk '{print $2}')

dir='/Users/ssq/Desktop/datasets/flora_test/'
bsz=32
worldsize=7
commfreq=7
backend='gloo'
model='resnet18'
dataset='cifar10'
localranks=(-1 0 0 1 2 1 2)

for val in $(seq 1 $worldsize)
do
  globalrank=$(($val-1))
  localrank=${localranks[$globalrank]}
  echo '###### going to launch training for global_rank '$globalrank' and local_rank '$localrank
  python3 -m src.flora.test.launch_hybridcomm --dir=$dir --bsz=$bsz --global-rank=$globalrank \
  --local-rank=$localrank --comm-freq=$commfreq --backend=$backend --model=$model --dataset=$dataset \
  --train-dir=$dir --test-dir=$dir &
  echo "going to sleep for 3 seconds..."
  sleep 3
done