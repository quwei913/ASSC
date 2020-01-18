#!/bin/bash

start=$1
end=$2
gpu=$3

for i in $(eval echo {$start..$end})
do
    CUDA_VISIBLE_DEVICES=$gpu python train.py --fold_idx $i --epochs 200 --batch_size 64
done
