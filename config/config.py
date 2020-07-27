import os
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()

_C.SYSTEM.NUM_GPUS = 1

# _C.SYSTEM.NUM_CPUS = 1
# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.Trainer = 'hv'
_C.MODEL.Name = 'hv'
_C.MODEL.lopt = []
_C.MODEL.wopt = []
_C.MODEL.copt = []
_C.MODEL.InputShape = []
_C.MODEL.OutputShape = []
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.Name = 'hv'
_C.DATASET.InputPath = ''
_C.DATASET.Image = ''
_C.DATASET.Label = ''
# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.BatchSize = 4
# # -----------------------------------------------------------------------------
# # Inference
# # -----------------------------------------------------------------------------
_C.INFERENCE = CN()


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
