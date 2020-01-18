#!/bin/bash
#for i in $(eval echo {0..19})
for i in $(eval echo {0..0})
do
        CUDA_VISIBLE_DEVICES=3 python train.py --fold_idx $i --epochs 1 --batch_size 64
done
