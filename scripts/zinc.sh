#!/bin/bash
# Note that num_reps = 1 because each run takes 48 hours
python run_model.py --dataset zinc --r 5 --shared --use_edge_attr --num_edge_encoder_layers 2 --num_layers 4 --num_decoder_layers 3 --num_reps 1 --norm BatchNorm1d --hidden_channels 256 --min_lr 1e-6 --batch_size 128 --num_epochs 2000