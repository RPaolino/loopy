from copy import deepcopy
import numpy as np
import os
from pympler import asizeof
import time 
import torch
import tqdm

from src.data import(
    best,
    build_dataset,
    build_loader,
    build_splits,
    freeze,
    get_evaluation_metric, 
    get_loss, 
    get_task,
    TSquared,
    custom_collate
)
from src.transforms import build_transform, r_stats
from src.nn import GNN
from src import (
    parse_args,
    seed_all,
    dict2yaml,
    save
)
from src.training import (
    get_scheduler,
    step_scheduler,
    training_step,
    eval_step
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 72.34 # Useful for BREC dataset
RESULTS_DIR = "results/"

def main(**passed_args):
    seed_all()
    # Getting command line args
    args = parse_args(**passed_args)
    if not args.config:
        dict2yaml(
            args, 
            os.path.join(
                RESULTS_DIR,
                f"{args.dataset}_{args.r}lwl_config.yaml"
            )
        )
    if (args.dataset.startswith("qm9") 
        or args.dataset.startswith("subgraphcount")):
        _, target = args.dataset.split("_")
        target = int(target)
    else:
        target = None
    # Loading the dataset.
    original_dataset = build_dataset(
        dataset_name=args.dataset,
        r=args.r,
        num_reps=args.num_reps
    )
    # Get task
    task = get_task(
        dataset_name=args.dataset
    )
    # Building splits: if one split is repeated, the seeds will be different
    splits, seeds = build_splits(
        dataset=original_dataset,
        dataset_name=args.dataset,
        num_reps=args.num_reps
    )
    # Build transforms
    transform = build_transform(
        dataset_name=args.dataset,
        r=args.r,
        target=target,
        lazy=args.lazy
    )   
    print("Transforms")
    for t in transform.transforms:
        print(f"\t{t}".expandtabs(4))
    time_start = time.time()
    # Freezing transforms.
    dataset = freeze(
        dataset=original_dataset,
        dataset_name=args.dataset,
        transform=transform,
        r=args.r,
        task=task
    )
    preprocessing_time = time.time()-time_start
    dataset_memory_usage = asizeof.asizeof(dataset)
    print(f"Dataset memory usage {dataset_memory_usage/1e9} GB")
    print("--")
    r_stats(dataset, r=args.r, lazy=args.lazy)
    print("--")
    # Get loss
    loss = get_loss(
        dataset_name=args.dataset
    )
    # Get evaluation metric
    metric_name, metric_fn = get_evaluation_metric(
        dataset_name=args.dataset
    )

    best_results = {
        f"loss_train": [], 
        f"loss_val": [],
        f"loss_test": [],
        f"{metric_name}_train": [], 
        f"{metric_name}_val": [],
        f"{metric_name}_test": [],
        f"training_time.mean":  [],
        f"eval_time.mean":  [],
        f"epochs": [],
        f"best_val_epoch": [],
    }
    # Starting the training
    for n_split, current_split in enumerate(splits, start=1):
        print("Split")
        print(
            f'\t{n_split}: '.expandtabs(4), 
            [len(idx) if idx is not None else None for idx in current_split],
            "seed", seeds[n_split-1]
        )
        seed_all(seeds[n_split-1])
        # Build train, val and test datasets
        datasets = [
            dataset[current_idx] if len(current_idx) else None for current_idx in current_split 
        ]
        # Build loaders
        loaders = build_loader(
            datasets=datasets,
            batch_size=args.batch_size,
            shuffle=[False]*3 if task=="T^2" else [True, False, False],
            seed=42
        )        
        # Build model
        if task == "num_identical_pairs":
            out_channels = 10
        elif task=="T^2":
            out_channels = 16
        else:
            out_channels = dataset.num_classes
        model = GNN(
            dataset=dataset,
            hidden_channels=args.hidden_channels,
            out_channels=out_channels,
            num_node_encoder_layers=args.num_node_encoder_layers, 
            num_edge_encoder_layers=args.num_edge_encoder_layers, 
            num_layers=args.num_layers, 
            num_decoder_layers=args.num_decoder_layers, 
            norm=args.norm,
            conv_dropout=args.conv_dropout, 
            dropout=args.dropout, 
            nonlinearity=args.nonlinearity,
            graph_pooling=args.pooling, 
            use_edge_attr=args.use_edge_attr,       
            r=args.r,
            shared=args.shared,
            residual=args.residual
        ).to(device)
        # and count his number of learnable parameters
        num_learnable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"N. learnable params: {num_learnable_params}")
        # Get optimizer
        optimizer = getattr(torch.optim, args.optimizer)(
            model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        # Get scheduler
        scheduler = get_scheduler(
            optimizer=optimizer,
            lr_scheduler=args.lr_scheduler,
            lr_scheduler_decay_steps=args.lr_scheduler_decay_steps,
            lr_scheduler_decay_rate=args.lr_scheduler_decay_rate,
            lr_scheduler_patience=args.lr_scheduler_patience
        )
        # Assuming these lists store your losses over epochs
        losses = {
            "train": [],
            "val": [],
            "test": []
        }
        evaluation_metrics = {
            "train": [],
            "val": [],
            "test": []
        }
        results = {
            "train": [],
            "val": [],
            "test": []
        }
        stats = {
            "training_time": [],
            "eval_time": [],
            "memory_allocated": [],
            "memory_reserved": [],
            "max_memory_allocated": [],
            "max_memory_reserved": []
        }
        # Begin training
        # Useful to standardize the feature
        mean = 0
        std = 1
        if (args.dataset.startswith("subgraphcount")
            or args.dataset.startswith("qm9")):
            train_idx = current_split[0]
            mean = dataset._data.y[train_idx].mean(0).item()
            std = dataset._data.y[train_idx].std(0).item()
            print("Standardizing:", f"mean={mean:.2E}", f"std={std:.2E}")
        
        progress = tqdm.tqdm(
            range(1, args.num_epochs+1), 
            desc="Training",
            disable=True if task=="num_identical_pairs" else False
        )
        for epoch in progress:
            time_start = time.time()
            training_step(
                dataset_name=args.dataset,
                mean=mean,
                std=std,
                model=model, 
                train_loader=loaders[0],
                optimizer=optimizer, 
                loss_fn=loss
            )
            stats["training_time"].append(time.time()-time_start)
            time_start = time.time()
            current_results = eval_step(
                dataset_name=args.dataset,
                mean=mean,
                std=std,
                model=model, 
                loaders=loaders,
                loss_fn=loss, 
                metric_name=metric_name, 
                metric_fn=metric_fn
            )
            stats["eval_time"].append(time.time()-time_start)
            # Appending results
            for split in ["train", "val", "test"]:
                results[split].append(
                    current_results[split]
                )
                losses[split].append(
                    current_results[split]["loss"]
                )
                evaluation_metrics[split].append(
                    current_results[split][metric_name]
                )
            stats["memory_allocated"].append(
                torch.cuda.memory_allocated()/1e9
            )
            stats["memory_reserved"].append(
                torch.cuda.memory_reserved()/1e9
            )
            stats["max_memory_allocated"].append(
                torch.cuda.max_memory_allocated()/1e9
            )
            stats["max_memory_reserved"].append(
                torch.cuda.max_memory_reserved()/1e9
            )
            torch.cuda.reset_peak_memory_stats()
            #Change description in console
            best_val_epoch = best(
                [val[metric_name] for val in results["val"]],
                metric_name=metric_name
            )
            if best_val_epoch+1==epoch:
                best_model = deepcopy(model)
            info = (
                f'Best {metric_name} at {best_val_epoch+1}: '
                + f'({results["train"][best_val_epoch][metric_name]:.2E}, '
                + f'{results["val"][best_val_epoch][metric_name]:.2E}, '
                + f'{results["test"][best_val_epoch][metric_name]:.2E})'
            )
            progress.set_description(info)
            # Step the scheduler
            step_scheduler(
                scheduler=scheduler, 
                lr_scheduler=args.lr_scheduler, 
                val_loss=current_results["val"]["loss"]
            )
            # Exit conditions
            if optimizer.param_groups[0]['lr'] < args.min_lr:
                break
        if args.dataset.startswith("brec"):
            filepath = f"equivalent/{args.dataset}_{args.r}lWL.csv"
            best_model.eval()
            y_pred = best_model(
                custom_collate(datasets[2])
            )
            distinct = TSquared(y_pred, None).item()
            reliable = TSquared(y_pred[np.random.randint(0, 2)::2], None).item()
            is_distinct = float(
                (distinct > THRESHOLD)
                and (not np.isclose(distinct, reliable, atol=1e-6))
            )
            is_reliable = float(reliable < THRESHOLD)
            #Removing old file and create a new one
            if n_split==1:
                if os.path.exists(filepath):
                    os.remove(filepath)
                else:
                    open(filepath, 'a').close()
            if not is_distinct or not is_reliable:
                with open(filepath, 'a') as file:
                    file.write(f"{original_dataset.subset[n_split-1]}\n")
            if "is_distinct" in best_results.keys():
                best_results["T2score"].append(distinct)
                best_results["T2score_reliability"].append(reliable)
                best_results["is_distinct"].append(is_distinct)
                best_results["is_reliable"].append(is_reliable)
            else:
                best_results["T2score"] = [distinct]
                best_results["T2score_reliability"] = [reliable]
                best_results["is_distinct"] = [is_distinct]
                best_results["is_reliable"] = [is_reliable]
            print(
                "T^2 score - avg distinct: ", 
                f"{np.mean(best_results['is_distinct']):.2E}±{np.std(best_results['is_distinct']):.2E}"
            )    
        best_val_epoch = best(
            [val[metric_name] for val in results["val"]],
            metric_name=metric_name
        )
        best_results["loss_train"].append(
            results["train"][best_val_epoch]['loss']
        )
        best_results["loss_val"].append(
            results["val"][best_val_epoch]['loss']
        )
        best_results["loss_test"].append(
            results["test"][best_val_epoch]['loss']
        )
        best_results[f"{metric_name}_train"].append(
            results["train"][best_val_epoch][metric_name]
        )
        best_results[f"{metric_name}_val"].append(
            results["val"][best_val_epoch][metric_name]
        )
        best_results[f"{metric_name}_test"].append(
            results["test"][best_val_epoch][metric_name]
        )
        best_results["epochs"].append(
            epoch
        )
        best_results["training_time.mean"].append(
            np.mean(stats["training_time"])
        )
        best_results["eval_time.mean"].append(
            np.mean(stats["eval_time"])
        )
        best_results["best_val_epoch"].append(
            int(best_val_epoch)
        )
        #Saving partial results
        avg_best_results = {}
        for key, value in best_results.items():
            if isinstance(value, list):
                if not np.any([np.isnan(v) for v in value]):
                    avg_best_results[f"{key}_mean"] = np.mean(value).item()
                    avg_best_results[f"{key}_std"] = np.std(value).item()
        # Print results to console
        if n_split==len(splits):
            log_text = "Avg. best results"
        else:
            log_text = (
               f"Partial avg. best results"
                + f' (fold from 1 to {n_split})'
            )
        # Save the results and their stats
        save(
            best_results, 
            filename=os.path.join(
                RESULTS_DIR,
                f"{args.dataset}_{args.r}lwl.csv"
            ),
            header=log_text
        )
        print(log_text)
        print(
            f'\t{"metric":^20} {"train":^17}\t{"val":^17}\t{"test":^17}'.expandtabs(4),
        )
        for fn in ["loss", metric_name]:
            log_text = f'\t{fn:^20} '
            for k in [f"{fn}_train", f"{fn}_val", f"{fn}_test"]:
                if k+"_mean" in avg_best_results.keys():
                    m = avg_best_results[k+"_mean"]
                    s = avg_best_results[k+"_std"]
                    log_text += f'{m:.2E}±{s:.2E}\t'
                else:
                    log_text += f'{"-":^17}\t'
            log_text = log_text.expandtabs(4)
            print(log_text)
        print("--")

        

if __name__ == "__main__":
    main()
    
