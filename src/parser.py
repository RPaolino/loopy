import argparse
import yaml

def parse_args(**passed_args):
    r"""
    Parse command line arguments. Allows either to load the args from a config file 
    (via "--config path/to/config.yaml") or to set directly the params via command line.
    """
    parser = argparse.ArgumentParser(description="An experiment.")
    parser.add_argument(
        "-v",
        "--verbose", 
        action="store_true",
        help="Print useful infos."
    )
    # Dataset
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="zinc",
        help="Dataset name (default: zinc)"
    )
    # Splits
    parser.add_argument(
        "--num_reps", 
        type=int, 
        default=5,
        help=(
            "Some datasets come with no split; a Stratified num_reps-fold "
            "cross validation is used to get random splits. Some datasets come "
            "with only one split; the split is then repeated num_reps times "
            "and the training happens on different seeds. (default: 5)"
            "For BREC, it specifies the number of permutations."
        )
    )
    # Loader
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64,
        help="Batch size (default: 64)"
    )
    # Transforms
    parser.add_argument(
        "--r", 
        type=int, 
        default=0,
        help="Neighbourhood order (default: 0)"
    )
    parser.add_argument(
        "--lazy", 
        action="store_true", 
        help=(
            "If True, it does not store all the paths, which can be computed "
            "as cyclic permutations of simple cycles. The advantage is less "
            "memory overload. The computation is then done in the forward, "
            "making each epoch a little slower."
        )
    )
    # GNN
    parser.add_argument(
        "--shared", 
        action="store_true",
        help="If True, one only layer process paths of different length."
    )
    parser.add_argument(
        "--nonlinearity", 
        type=str, 
        default="relu",
        help="Activation function (default: relu)"
    )
    parser.add_argument(
        "--norm", 
        type=str, 
        choices=["BatchNorm1d", "LayerNorm", "Identity"],
        default="Identity",
        help=f"Graph normalization (default `Identity`)"
    )
    parser.add_argument(
        "--conv_dropout", 
        type=float, 
        default=0.0,
        help="Dropout (after convolution) rate (default: 0.0)"
    )
    parser.add_argument(
        "--dropout", 
        type=float, 
        default=0.0,
        help="Dropout rate (default: 0.0)"
    )
    parser.add_argument(
        "--hidden_channels", 
        type=int, 
        default=64,
        help="Num. of hidden channels (default: 64)"
    )
    parser.add_argument(
        "--num_layers", 
        type=int, 
        default=3,
        help="Num. of layers (default: 3)"
    )
    parser.add_argument(
        "--num_decoder_layers", 
        type=int, 
        default=2,
        help="Num. of decoding MLP layers that predict the embedding (default: 2)"
    )
    parser.add_argument(
        "--num_node_encoder_layers", 
        type=int, 
        default=2,
        help="Num. of MLP layers encoding node features (default: 2)"
    )
    parser.add_argument(
        "--num_edge_encoder_layers", 
        type=int, 
        default=0,
        help=(
            "Num. of MLP layers encoding edge features. Note that you need to "
            "call --use_edge_attr if you want to use the edge attribs. during "
            "training. (default: 0). This choice because you may want to use"
            "edge attribs. without encoding them first."
        )
    )
    parser.add_argument(
        "--use_edge_attr", 
        action="store_true", 
        help="If the model should use edge attributes"
    )  
    parser.add_argument(
        "--residual", 
        action="store_true", 
        help="If the model should have a residual connection"
    )  
    parser.add_argument(
        "--pooling", 
        type=str, 
        default="sum",
        choices=["sum", "mean", "max", "min"],
        help="Graph pooling operation (default: sum)"
    )
    
    # Optimizer
    parser.add_argument(
        "--optimizer", 
        type=str, 
        default="Adam",
        help="Optimizer to use (default: Adam)"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-3,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.,
        help="Optimizer weight decay (default: 0.)"
    )
    # LR scheduler
    parser.add_argument(
        "--lr_scheduler", 
        type=str, 
        default="ReduceLROnPlateau",
        choices=["", "StepLR", "ReduceLROnPlateau"],
        help="Scheduler (default ReduceLROnPlateau)"
    )
    parser.add_argument(
        "--lr_scheduler_decay_rate", 
        type=float, 
        default=0.5,
        help="Strength of lr decay (default: 0.5)"
    )
    parser.add_argument(
        "--lr_scheduler_decay_steps", 
        type=int, 
        default=50,
        help="For StepLR, number of epochs between lr decay (default: 50)"
    )
    parser.add_argument(
        "--min_lr", 
        type=float, 
        default=1e-5,
        help="For ReduceLROnPlateau, mininum learnin rate (default: 1e-5)"
    )
    parser.add_argument(
        "--lr_scheduler_patience", 
        type=int, 
        default=50,
        help="For ReduceLROnPlateau, number of epochs without improvement"
    )
    # Miscellanea
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=1000,
        help="Num. of epochs (default: 1000)"
    )  
    # Config file to load
    parser.add_argument(
        "--config", 
        dest="config", 
        type=str,
        default="",
        help=f"Path to a config file that should be used for this experiment. "
            + f"CANNOT be combined with explicit arguments"
    )
    args = parser.parse_args()
    # Load partial args instead of command line args (if they are given)
    if passed_args:
        for key, value in passed_args.items():
            args.__dict__[key] = value
    args.__dict__["dataset"] = args.__dict__["dataset"].lower()
    # https://codereview.stackexchange.com/a/79015
    # If a config file is provided, write it"s values into the arguments
    if args.config:
        with open(args.config, "r") as f:
            config_file_args = yaml.safe_load(f)
            for key, value in config_file_args.items():
                    args.__dict__[key] = value
    return args
