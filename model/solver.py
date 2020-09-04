from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, StepLR
from torch.optim import SGD, Adam


def build_scheduler(cfg, optimizer):
    if cfg.SOLVER.LR_SCHEDULER == 'ReduceOnPlateau':
        return ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=cfg.SOLVER.GAMMA, patience=10)
    if cfg.SOLVER.LR_SCHEDULER == 'MultiStepLR':
        return MultiStepLR(optimizer=optimizer, milestones=cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA)


def build_optimizer(cfg, model):
    if cfg.SOLVER.OPTIMIZER == 'Adam':
        return Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    if cfg.SOLVER.OPTIMIZER == 'SGD':
        return SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR)


def build_solver(cfg, model):
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_scheduler(cfg, optimizer)
    return optimizer, lr_scheduler
