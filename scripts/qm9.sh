#!/bin/bash
python run_model.py --dataset qm9_0 --r 5 --use_edge_attr --num_edge_encoder_layers 2 --num_layers 4 --num_reps 4 --norm BatchNorm1d --num_epochs 400 --lr_scheduler_decay_rate 0.9 --lr_scheduler_patience 10
python run_model.py --dataset qm9_1 --r 5 --use_edge_attr --num_edge_encoder_layers 2 --num_layers 5 --num_reps 4 --norm BatchNorm1d --num_epochs 400 --lr_scheduler_decay_rate 0.9 --lr_scheduler_patience 10
python run_model.py --dataset qm9_2 --r 5 --use_edge_attr --num_edge_encoder_layers 2 --num_layers 5 --num_reps 4 --norm BatchNorm1d --num_epochs 400 --lr_scheduler_decay_rate 0.9 --lr_scheduler_patience 10
python run_model.py --dataset qm9_3 --r 5 --use_edge_attr --num_edge_encoder_layers 2 --num_layers 4 --num_reps 4 --norm BatchNorm1d --num_epochs 400 --lr_scheduler_decay_rate 0.9 --lr_scheduler_patience 10
python run_model.py --dataset qm9_4 --r 5 --use_edge_attr --num_edge_encoder_layers 2 --num_layers 4 --num_reps 4 --norm BatchNorm1d --num_epochs 400 --lr_scheduler_decay_rate 0.9 --lr_scheduler_patience 10
python run_model.py --dataset qm9_5 --r 5 --use_edge_attr --num_edge_encoder_layers 2 --num_layers 4 --num_reps 4 --norm BatchNorm1d --num_epochs 400 --lr_scheduler_decay_rate 0.9 --lr_scheduler_patience 10
python run_model.py --dataset qm9_6 --r 5 --use_edge_attr --num_edge_encoder_layers 2 --num_layers 3 --num_reps 4 --norm BatchNorm1d --num_epochs 400 --lr_scheduler_decay_rate 0.9 --lr_scheduler_patience 10
python run_model.py --dataset qm9_7 --r 5 --use_edge_attr --num_edge_encoder_layers 2 --num_layers 6 --num_reps 4 --norm BatchNorm1d --num_epochs 400 --lr_scheduler_decay_rate 0.9 --lr_scheduler_patience 10
python run_model.py --dataset qm9_8 --r 5 --use_edge_attr --num_edge_encoder_layers 2 --num_layers 5 --num_reps 4 --norm BatchNorm1d --num_epochs 400 --lr_scheduler_decay_rate 0.9 --lr_scheduler_patience 10
python run_model.py --dataset qm9_9 --r 5 --use_edge_attr --num_edge_encoder_layers 2 --num_layers 6 --num_reps 4 --norm BatchNorm1d --num_epochs 400 --lr_scheduler_decay_rate 0.9 --lr_scheduler_patience 10
python run_model.py --dataset qm9_10 --r 5 --use_edge_attr --num_edge_encoder_layers 2 --num_layers 4 --num_reps 4 --norm BatchNorm1d --num_epochs 400 --lr_scheduler_decay_rate 0.9 --lr_scheduler_patience 10
python run_model.py --dataset qm9_11 --r 5 --use_edge_attr --num_edge_encoder_layers 2 --num_layers 4 --num_reps 4 --norm BatchNorm1d --num_epochs 400 --lr_scheduler_decay_rate 0.9 --lr_scheduler_patience 10
