import os
from ..config import get_cfg_defaults
import argparse
import torch

def get_args():
    r"""Get args from command lines.
    """
    parser = argparse.ArgumentParser(description="Model Training & Inference")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    parser.add_argument('--inference', action='store_true', help='inference mode')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to load the checkpoint')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    # configurations
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    mode = 'test' if args.inference else 'train'
