#!/bin/bash
for dataset in exp cexp
do
    for r in 0 1 2 3 4 5 6
    do
        python run_model.py --dataset $dataset --r $r --num_reps 4
    done
done
for r in 0 1 2 3 4 5 6
    do
        python run_model.py --dataset csl --r $r
    done
done