from torch.utils.data import Dataset
from dataclasses import dataclass, fields
from torch.utils.data import random_split
import os
from .dataset import ImageTripletDataset


@dataclass
class DatasetSplit:
    train: Dataset
    val: Dataset
    test: Dataset

    @staticmethod
    def create_train_val_test_split(
        full_dataset: ImageTripletDataset, test_fraction=0.25, val_fraction=0.2
    ):
        train_val_dataset, test_dataset = DatasetSplit.split_dataset(
            full_dataset, test_fraction
        )
        train_dataset, val_dataset = DatasetSplit.split_dataset(
            train_val_dataset, val_fraction
        )
        dataset_split = DatasetSplit(
            train=train_dataset, val=val_dataset, test=test_dataset
        )
        dataset_split.train = full_dataset.create_slice(dataset_split.train.indices)
        dataset_split.val = full_dataset.create_slice(dataset_split.val.indices)
        dataset_split.test = full_dataset.create_slice(dataset_split.test.indices)
        return dataset_split

    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name) for field in fields(self.__class__)
        }

    @staticmethod
    def split_dataset(dataset, split_fraction=0.25):
        second_size = int(split_fraction * len(dataset))
        first_set, second_set = random_split(
            dataset, [len(dataset) - second_size, second_size]
        )
        return first_set, second_set
