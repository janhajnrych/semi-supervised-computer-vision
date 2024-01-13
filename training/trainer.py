import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
from abc import abstractmethod
from .aspect import Aspect


class Trainer(Aspect):
    def __init__(self, model: nn.Module, batch_size=64):
        super().__init__(model, batch_size=batch_size)
        self.optimizer = optim.Adam(model.parameters())
        self.criterion = nn.TripletMarginWithDistanceLoss(
            distance_function=nn.PairwiseDistance(), margin=1
        )

    def train_epoch(self, dataset):
        self.model.train()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        losses = []
        for batch_idx, (images, labels, ids) in tqdm(
            enumerate(loader), total=len(loader), desc="training"
        ):
            loss, embeddings = self.train_batch(images, labels, ids)
            losses.append(loss)
        return losses

    @abstractmethod
    def train_batch(self, images, labels, idx):
        pass

    @abstractmethod
    def get_transform(self, input_size):
        pass
