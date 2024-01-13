import numpy as np
import torch
import os
from dataset import DatasetSplit
from .trainer import Trainer
from .validator import Validator
from .tester import Tester
from dataclasses import dataclass
import logging


class Session:
    class SessionLoadError(RuntimeError):
        pass

    @dataclass
    class SessionStrategy:
        trainer: Trainer
        validator: Validator
        tester: Tester

    @dataclass
    class ReducedPoint:
        mean: float
        std: float

        def __str__(self):
            return f"{self.mean:.3g}(+-{self.std:.3g})"

    def __init__(self, datasets: DatasetSplit, strategy: SessionStrategy):
        self.datasets = datasets
        self.strategy = strategy
        self._history = []

    def test(self):
        batch_scores = self.strategy.tester.test_epoch(self.datasets.val)
        return self.build_score_stats(batch_scores)

    @staticmethod
    def reduce(data: list):
        return Session.ReducedPoint(mean=float(np.mean(data)), std=float(np.std(data)))

    @staticmethod
    def build_score_stats(scores):
        stats = {}
        if len(scores) == 0:
            return stats
        columns = set(scores[0].keys())
        for c in columns:
            stats[c] = Session.reduce([i[c] for i in scores])
        return stats

    @staticmethod
    def _flatten_history_item(element):
        out = {}
        for k, v in element.items():
            if isinstance(v, Session.ReducedPoint):
                out[k + "_mean"] = v.mean
                out[k + "_std"] = v.std
            else:
                out[k] = v
        return out

    @property
    def history(self):
        return [self._flatten_history_item(i) for i in self._history]

    @staticmethod
    def _is_trackable(item):
        return (
            isinstance(item, int)
            or isinstance(item, float)
            or isinstance(item, Session.ReducedPoint)
        )

    def train(self, n_epochs=30, start_epoch=0):
        for epoch in range(start_epoch, n_epochs + start_epoch):
            logging.debug(f"start epoch {epoch}")
            train_losses = self.strategy.trainer.train_epoch(self.datasets.train)
            val_losses, val_scores = self.strategy.validator.validate_epoch(
                self.datasets.val
            )
            stats = {
                "val_loss": Session.reduce(val_losses),
                "train_loss": Session.reduce(train_losses),
            }
            stats.update(self.build_score_stats(val_scores))
            stats["epoch"] = epoch + 1
            trackable = {k: v for k, v in stats.items() if self._is_trackable(v)}
            self._history.append(trackable)
            logging.debug(f"stats: {stats}")
            yield epoch
            logging.debug(f"finished epoch {epoch}")

    def save_model(self, path):
        model = self.strategy.trainer.model
        torch.save(model.state_dict(), path)

    def load_model(self, path):
        model = self.strategy.trainer.model
        model.load_state_dict(torch.load(path))
