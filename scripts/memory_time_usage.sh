#!/bin/bash
for r in 0 1 2 3 4 5
do
    python run_model.py --dataset qm9_0 --r $r --use_edge_attr --num_edge_encoder_layers 2 --num_layers 5 --num_reps 1 --norm BatchNorm1d --num_epochs 20 --lr_scheduler_decay_rate 0.9 --lr_scheduler_patience 10
done