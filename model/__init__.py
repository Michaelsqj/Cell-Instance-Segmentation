from .hover_sa import HoverNet
from .utils import *

model_zoo = {'hv': HoverNet}


def build_model(name):
    return model_zoo[name]
