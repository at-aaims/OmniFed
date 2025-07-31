#!/bin/sh

cd ../

kafkadir='/ccsopen/home/ssq/kafka_2.12-3.2.0'
datadir='/ccsopen/home/ssq/datasets/'
logdir='/ccsopen/home/ssq/datasets/'
kafkahost='127.0.0.1'
kafkaport=9092
dataset='cifar10'
totalclients=1
streamrate=32

# run subscriber here
for val in $(seq 1 $totalclients)
do
  rank=$(($val-1))--kafka-dir
  topic='client-'$rank
  python3 -m src.flora.stream_simulation.run_client_subscriber --kafka-dir=$kafkadir --data-dir=$datadir \
  --log-dir=$logdir --kafka-host=$kafkahost --kafka-port=$kafkaport --kafka-topic=$topic --client-id=$rank &
  echo "going to sleep subscriber creation for 2 seconds..."
  sleep 2
done

# run publisher here
echo "going to launch Publisher server..."
python3 -m src.flora.stream_simulation.run_server_publisher --kafka-dir=$kafkadir --dataset=$dataset \
--kafka-host=$kafkahost --kafka-port=$kafkaport --total-clients=$totalclients --stream-rate=$streamrate &