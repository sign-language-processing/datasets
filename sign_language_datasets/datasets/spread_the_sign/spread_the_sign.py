"""Spreadthesign"""

import csv
from os import path


import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.io.gfile import GFile

from pose_format import Pose

from ..warning import dataset_warning
from ...datasets.config import SignDatasetConfig

_DESCRIPTION = """
SpreadTheSign2 is a notable multilingual dictio- nary containing around 23,000 words with up to 41 different spoken-sign language pairs and more than 600,000 videos in total.
"""

_CITATION = """
"""

_POSE_HEADERS = {"holistic": path.join(path.dirname(path.realpath(__file__)), "holistic.poseheader")}

_KNOWN_SPLITS = {"1.0.0-uzh": path.join(path.dirname(path.realpath(__file__)), "splits/1.0.0-uzh")}


class SpreadTheSign(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Spreadthesign dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    BUILDER_CONFIGS = [SignDatasetConfig(name="default", include_pose="holistic")]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {
            "id": tfds.features.Text(),
            "text": tfds.features.Text(),
            "sign_language": tfds.features.Text(),
            "spoken_language": tfds.features.Text(),
            "pose_path": tfds.features.Text(),
            "pose_length": tf.float32,
        }

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="https://www.spreadthesign.com/",
            supervised_keys=None,
            citation=_CITATION,
        )

    def _load_split_ids(self, split: str):
        split_dir = _KNOWN_SPLITS[self._builder_config.extra["split"]]

        with open(path.join(split_dir, f"{split}.txt")) as f:
            ids = []
            for line in f:
                id = line.rstrip("\n")
                ids.append(id)

        return ids

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dataset_warning(self)

        pose_dir = self._builder_config.extra["pose_dir"]

        if "split" in self._builder_config.extra:
            train_args = {"pose_dir": pose_dir, "ids": self._load_split_ids("train")}
            val_args = {"pose_dir": pose_dir, "ids": self._load_split_ids("val")}
            test_args = {"pose_dir": pose_dir, "ids": self._load_split_ids("test")}

            return [
                tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs=train_args),
                tfds.core.SplitGenerator(name=tfds.Split.VALIDATION, gen_kwargs=val_args),
                tfds.core.SplitGenerator(name=tfds.Split.TEST, gen_kwargs=test_args),
            ]
        else:
            return [tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"pose_dir": pose_dir})]

    def _generate_examples(self, pose_dir: str, ids: list = []):
        """Yields examples."""

        with GFile(self._builder_config.extra["csv_path"], "r") as csv_file:
            csv_data = csv.reader(csv_file, delimiter=",")
            next(csv_data)  # Ignore the header

            for i, row in enumerate(csv_data):
                datum = {"id": str(i), "text": row[3], "sign_language": row[1], "spoken_language": row[2]}

                if len(ids) > 0 and (datum["id"] not in ids):
                    continue

                if self.builder_config.include_pose is not None:
                    if self.builder_config.include_pose == "holistic":
                        mediapipe_path = path.join(pose_dir, row[0])

                        if path.exists(mediapipe_path):
                            datum["pose_path"] = mediapipe_path
                            with open(mediapipe_path, "rb") as f:
                                pose = Pose.read(f.read())
                                datum["pose_length"] = pose.body.data.shape[0]

                            yield datum["id"], datum
