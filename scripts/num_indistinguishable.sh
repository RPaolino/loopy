#!/bin/bash
for dataset in cospectral10 graph8c exp_iso
do
    for r in 0 1 2 3
    do
        python run_model.py --dataset $dataset --r $r --num_epochs 1 --num_reps 100
    done
done
for dataset in sr16622
do
    for r in 0 1 2 3 4
    do
        python run_model.py --dataset $dataset --r $r --num_epochs 1 --num_reps 100
    done
done