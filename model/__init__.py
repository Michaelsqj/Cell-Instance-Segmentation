from .hover_sa import HoverNet
from .criterion import Criterion
model_zoo = {'hv': HoverNet}


def build_model(name):
    return model_zoo[name]
