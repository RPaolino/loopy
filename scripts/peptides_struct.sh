#!/bin/bash
python run_model.py --dataset peptides_struct --r 7 --use_edge_attr --num_edge_encoder_layers 2 --num_layers 4 --num_reps 4 --norm BatchNorm1d --num_epochs 400
