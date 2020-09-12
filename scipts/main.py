from Segmentation.config import get_cfg_defaults
import argparse
import torch
from Segmentation.engine import trainer_zoo
from Segmentation.logs import build_logger


def get_args():
    r"""Get args from command lines.
    """
    parser = argparse.ArgumentParser(description="Model Training & Inference")
    parser.add_argument('--config', type=str, help='configuration file (yaml)')
    parser.add_argument('--inference', action='store_true', help='inference mode')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to load the checkpoint')
    parser.add_argument('--logs', type=str, default='logs', help='log file name')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    logger = build_logger(args.logs)
    # configurations
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('device', str(device))
    mode = 'test' if args.inference else 'train'
    trainer = trainer_zoo[cfg.MODEL.TRAINER](cfg, mode, device)
    if mode == 'train':
        trainer.train()
    else:
        trainer.test()
