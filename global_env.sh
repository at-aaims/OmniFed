#!/bin/sh

# for skewed and unbalanced data partitioning schemes in src/omnifed/data/skewed
export OMNIFED_PARTITION_SEED=123

# for testing w/ C4 dataset
export OMNIFED_TOKENIZER_DIR=""
export OMNIFED_C4_DISK=""
export OMNIFED_DATA_DIR=""