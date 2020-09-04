from .hover_sa import HoverNet
from .criterion import Criterion
from .solver import build_solver
from .visualization import visualizer
model_zoo = {'hv': HoverNet}


def build_model(name):
    return model_zoo[name]
