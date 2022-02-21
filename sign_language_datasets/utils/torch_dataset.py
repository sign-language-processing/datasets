import tensorflow_datasets as tfds
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig

import itertools
import numpy as np

import torch


class TFDSTorchDataset(torch.utils.data.IterableDataset):
    def __init__(self, tf_dataset):
        self.dataset = tf_dataset

    def __len__(self):
        return len(self.dataset)

    def torch_obj(self, obj):
        if isinstance(obj, dict):
            return {k: self.torch_obj(v) for k, v in obj.items()}
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        return obj

    def __iter__(self):
        for datum in self.dataset:
            yield self.torch_obj(tfds.as_numpy(datum))


if __name__ == "__main__":
    # Fast download and load dataset using TFDS
    config = SignDatasetConfig(name="holistic-poses", version="1.0.0", include_video=False, include_pose="holistic")
    dicta_sign = tfds.load(name='dicta_sign', builder_kwargs={"config": config})

    # Convert to torch dataset
    train_dataset = TFDSTorchDataset(dicta_sign["train"])

    for datum in itertools.islice(train_dataset, 0, 10):
        print(datum)
