import torch.nn as nn
from training import Trainer, Validator, Tester, Aspect
import torchvision.transforms as transforms
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


class TripletStrategy(Aspect):
    def __init__(self, model: nn.Module, batch_size=64):
        super().__init__(model, batch_size=batch_size)
        self.distance = nn.PairwiseDistance()
        self.criterion = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance, margin=1
        )

    def get_loss(self, embeddings):
        return self.criterion(embeddings[0], embeddings[1], embeddings[2])

    def forward(self, images):
        tensors = [i.unsqueeze(1).to(self.device).float() for i in images]
        return [self.model(tensor) for tensor in tensors]

    @staticmethod
    def get_data_transform(input_size, margin=8):
        return transforms.Compose(
            [
                transforms.Resize((input_size + margin, input_size + margin)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((input_size, input_size)),
                transforms.Grayscale(),
                transforms.Normalize(0.5, 0.5),
            ]
        )


class TripletTrainer(Trainer, TripletStrategy):
    def __init__(self, model: nn.Module, batch_size=64):
        super().__init__(model, batch_size=batch_size)
        self.distance = nn.PairwiseDistance()

    def train_batch(self, images, labels, idx):
        self.optimizer.zero_grad()
        embeddings = self.forward(images)
        loss = self.get_loss(embeddings)
        loss.backward()
        self.optimizer.step()
        return loss.item(), embeddings

    def get_transform(self, input_size):
        return self.get_data_transform(input_size)


class AnalyzerStrategy(TripletStrategy):
    def _aggregate(self, embeddings):
        dist_pos = self.distance(embeddings[0], embeddings[1]).detach().cpu().numpy()
        dist_neg = self.distance(embeddings[0], embeddings[2]).detach().cpu().numpy()
        return dist_pos, dist_neg

    def analyze(self, embeddings):
        dist_pos, dist_neg = self._aggregate(embeddings)
        true_labels = [0]
        while len(set(list(true_labels))) != 2:
            true_labels = np.random.randint(2, size=dist_pos.shape[0])
        n = true_labels.shape[0]
        scores = np.array(
            [dist_pos[i] if true_labels[i] == 0 else dist_neg[i] for i in range(n)]
        )
        pred_labels = np.array(
            [
                dist_pos[i] < dist_neg[i]
                if true_labels[i] == 1
                else dist_neg[i] < dist_pos[i]
                for i in range(n)
            ]
        )
        return true_labels, scores, pred_labels

    @staticmethod
    def flatten(batch_data_list, key_list):
        data = {k: list() for k in key_list}
        for batch_data in batch_data_list:
            for k in key_list:
                data[k].extend(batch_data[k])
        for k in key_list:
            data[k] = np.array(data[k])
        return data


class TripletValidator(Validator, AnalyzerStrategy):
    def __init__(self, model: nn.Module, batch_size=64):
        super().__init__(model, batch_size=batch_size)
        self.distance = torch.nn.PairwiseDistance()

    def validate_batch(self, images, labels):
        embeddings = self.forward(images)
        true_labels, scores, pred_labels = self.analyze(embeddings)
        loss = self.get_loss(embeddings).item()
        metrics = {
            "roc_auc_batch": roc_auc_score(true_labels, scores),
        }
        return loss, metrics

    def get_transform(self, input_size):
        return self.get_data_transform(input_size)


class TripletTester(Tester, AnalyzerStrategy):
    def __init__(self, model: nn.Module, batch_size=64):
        super().__init__(model, batch_size=batch_size)
        self.distance = torch.nn.PairwiseDistance()

    def test_batch(self, images, labels):
        embeddings = self.forward(images)
        true_labels, scores, pred_labels = self.analyze(embeddings)
        return {"roc_auc": roc_auc_score(true_labels, scores)}

    def get_transform(self, input_size):
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.Grayscale(),
                transforms.Normalize(0.5, 0.5),
            ]
        )
