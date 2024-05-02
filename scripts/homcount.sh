#!/bin/bash
for cycle_length in 3 4 5 6
do
    for r in 0 1 2 3 4
    do
        python run_model.py --dataset subgraphcount_$cycle_length --r $r
    done
done