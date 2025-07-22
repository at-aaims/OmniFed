#!/bin/bash

clear

# Clear Hydra outputs directory
rm -rf ./outputs

configs=(
    test_mnist_centralized_torchdist
    test_mnist_centralized_grpc
    test_mnist_multigroup
)

for config in "${configs[@]}"; do

    ./main.sh --config-name "$config"

    echo
done
