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
_C.MODEL.TRAINER = 'hv'
_C.MODEL.NAME = 'hv'
_C.MODEL.LOPT = []
_C.MODEL.WOPT = []
_C.MODEL.COPT = []
_C.MODEL.LOSS_WEIGHT = []
_C.MODEL.INPUT_SHAPE = []
_C.MODEL.OUTPUT_SHAPE = []
_C.MODEL.REJECT_SIZE = 10
_C.MODEL.REJECT_P = 1.0
_C.MODEL.VIS_ITER = 10
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = 'hv'
_C.DATASET.INPUT_PATH = ''
_C.DATASET.OUTPUT_PATH = ''
_C.DATASET.IMAGE = ''
_C.DATASET.LABEL = ''

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.BATCH_SIZE = 4
_C.SOLVER.ITERATION_TOTAL = 600
_C.SOLVER.OPTIMIZER = 'Adam'
_C.SOLVER.LR_SCHEDULER = 'ReduceOnPlateau'
_C.SOLVER.GAMMA = 0.5
_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.STEPS = [50, 100]
_C.SOLVER.ITERATION_SAVE = 100
_C.SOLVER.ITERATION_RESTART = False
# # -----------------------------------------------------------------------------
# # Inference
# # -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.IMAGE = ''
_C.INFERENCE.STRIDE = 64
_C.INFERENCE.OUTPUT_PATH = ''


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
