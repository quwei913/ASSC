#!/bin/bash
set -x
./batch_train.sh 0 9 1 > test_9.out &
./batch_train.sh 10 19 3 > test_19.out &
