import torch.nn as nn
import numpy as np
from dataclasses import dataclass


class Aspect:
    @dataclass
    class ReducedLoss:
        mean: np.array
        std: np.array

    def __init__(self, model: nn.Module, batch_size=64):
        self.model = model
        self.device = next(model.parameters()).device
        self.batch_size = batch_size

    @staticmethod
    def reduce(data: list):
        return Aspect.ReducedLoss(mean=np.mean(data), std=np.std(data))
