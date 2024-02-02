#!/bin/bash
for dataset in exp cexp csl
do
    for r in 0 1 2 3 4 5 6
    do
        python run_model.py --dataset $dataset --r $r
    done
done