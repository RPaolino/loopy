#!/bin/bash
python run_model.py --dataset zinc_subset --r 5 --shared --use_edge_attr --num_edge_encoder_layers 2 --num_reps 4 --norm BatchNorm1d --hidden_channels 128