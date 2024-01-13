from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
from abc import abstractmethod
from .aspect import Aspect


class Validator(Aspect):
    def __init__(self, model: nn.Module, batch_size=64):
        super().__init__(model, batch_size=batch_size)
        self.criterion = nn.TripletMarginWithDistanceLoss(
            distance_function=nn.PairwiseDistance(), margin=1
        )

    @abstractmethod
    def validate_batch(self, images, labels):
        pass

    def validate_epoch(self, dataset):
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        losses = []
        batch_scores = []
        for batch_idx, (images, labels, ids) in tqdm(
            enumerate(loader), total=len(loader), desc="validation"
        ):
            val_loss, score_dict = self.validate_batch(images, labels)
            losses.append(val_loss)
            batch_scores.append(score_dict)
        return losses, batch_scores

    @abstractmethod
    def get_transform(self, input_size):
        pass
