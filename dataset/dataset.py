from torch.utils.data import Dataset
import os
import random
from dataclasses import dataclass
from tqdm import tqdm
import torchvision.transforms as transforms
import logging
from PIL import Image


class ImageTripletDataset(Dataset):
    size = (128, 128)

    @dataclass
    class ImageData:
        path: str
        label: int

    @dataclass
    class ItemData:
        images: tuple
        labels: tuple | None | int
        ids: tuple

    def __init__(self, transform=transforms.ToTensor(), items=None):
        self._transform = transform
        if items is None:
            self.data = []
        elif isinstance(items, list):
            self.data = items
        else:
            raise ValueError("invalid dataset items")

    def _get_item(self, idx):
        anchor = self.data[idx]
        n = len(self.data)
        positive_indices = list(
            filter(lambda i: self.data[i].label == anchor.label, range(n))
        )
        positive_index = random.choice(positive_indices)
        positive = self.data[positive_index]
        negative_indices = list(
            filter(lambda i: self.data[i].label != anchor.label, range(n))
        )
        negative_index = random.choice(negative_indices)
        negative = self.data[negative_index]
        items = [anchor, positive, negative]
        images = [self._transform(Image.open(item.path)).squeeze(0) for item in items]
        labels = [1, 0]
        return self.ItemData(
            images=tuple(images),
            labels=tuple(labels),
            ids=(idx, positive_index, negative_index),
        )

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transforms.ToTensor()
        if transform is not None:
            self._transform = transforms.Compose([self._transform, transform])

    def load_from_dir(self, root_dir):
        subdir_list = os.listdir(root_dir)
        for i in tqdm(
            range(len(subdir_list)), desc=f"loading {os.path.basename(root_dir)}"
        ):
            subdir_path = os.path.join(root_dir, subdir_list[i])
            if not os.path.isdir(subdir_path):
                continue
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                self.data.append(self.ImageData(path=file_path, label=i))
        self.data = list(sorted(self.data, key=lambda item: item.path))
        logging.info(f"loaded {os.path.abspath(root_dir)} n={len(self.data)}")

    def create_slice(self, indices):
        return ImageTripletDataset(
            transform=transforms, items=[self.data[i] for i in indices]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self._get_item(idx)
        return item.images, item.labels, item.ids
