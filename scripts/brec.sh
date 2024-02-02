#!/bin/bash
for dataset in basic extension regular
do
    for r in 1 2 3 4
    do
        python run_model.py --dataset brec_$dataset --r $r --num_layers 5 --hidden_channels 32 --lr 1e-4 --num_epochs 40 --shared --lazy --num_reps 32
    done
done
for dataset in 4vtx
do
    for r in 1
    do
        python run_model.py --dataset brec_$dataset --r $r --num_layers 5 --hidden_channels 32 --lr 1e-4 --num_epochs 40 --shared --lazy --num_reps 32
    done
done
for dataset in dr
do
    for r in 1 2 3
    do
        python run_model.py --dataset brec_$dataset --r $r --num_layers 5 --hidden_channels 32 --lr 1e-4 --num_epochs 40 --shared --lazy --num_reps 32
    done
done
for dataset in str cfi
do
    for r in 1 2 3 4
    do
        python run_model.py --dataset brec_$dataset --r $r --num_layers 5 --hidden_channels 32 --lr 1e-4 --num_epochs 40 --shared --lazy --batch_size 4 --num_reps 32
    done
done