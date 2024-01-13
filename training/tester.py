from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
from abc import abstractmethod
from .aspect import Aspect


class Tester(Aspect):
    def __init__(self, model: nn.Module, batch_size=64):
        super().__init__(model, batch_size=batch_size)

    @abstractmethod
    def test_batch(self, images, labels):
        pass

    def test_epoch(self, dataset):
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        batch_scores = []
        for batch_idx, (images, labels, ids) in tqdm(
            enumerate(loader), total=len(loader), desc="testing"
        ):
            batch_scores.append(self.test_batch(images, labels))
        return batch_scores

    @abstractmethod
    def get_transform(self, input_size):
        pass
