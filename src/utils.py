import numpy as np
import os
import random
import torch
import yaml

bcolors = {
    'prev_line': '\033[F',
    'clear_line': '\033[K',
    'grey' : '\033[90m',
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
    'white': '\033[97m',
    'black' : '\033[90m',
    'default' : '\033[99m',
    'bold': '\033[1m',
    'underline': '\033[4m',
    'end': '\033[0m'
}

def blue(text):
    return f'{bcolors["blue"]}{text}{bcolors["end"]}'

def yellow(text):
    return f'{bcolors["yellow"]}{text}{bcolors["end"]}'

def seed_all(seed=1000):
    r'''
    Manually set the seeds for torch, numpy
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.set_num_threads(1)

def dict2yaml(args, filename):
    r"""
    Store ``metrics`` and save it at the location specified by `filename`.
    
    Args:
        args (dict): dict to store.
        filenam (str): name of the file where to store `args`.
    """
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)

def save(metrics, filename, header=""):
    r"""
    Save the ``metrics`` at the location specified by `filename`. Moreover,
    it appends mean and std as last rows in the file.
    
    Args:
        metrics (dict): dict of metrics to plot. Keys are used as header.
        filenam (str): name of the file where to store the plot
    """
    content2save = np.hstack(
        [np.array(l).reshape(-1, 1) for l in metrics.values()]
    )
    content2save = np.vstack(
        [content2save, content2save.mean(0), content2save.std(0)]
    )
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    metrics_fmt = {
        k:"%d" if k in ["is_distinct", "is_reliable", "epochs", "best_val_epoch"] else "%.16f" for k in metrics.keys()
    }
    with open(filename, 'w') as outfile:
        np.savetxt(
            outfile, 
            content2save, 
            header=(
                header + "\n" * (len(header)>0) 
                + "Last two rows are the mean and std\n"
                + "\t".join(list(metrics.keys()))
            ),
            delimiter="\t",
            fmt=list(metrics_fmt.values())
        )   