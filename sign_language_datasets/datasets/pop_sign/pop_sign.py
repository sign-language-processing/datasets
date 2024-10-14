"""PopSign ASL v1.0: An Isolated American Sign Language Dataset Collected via Smartphones"""

import csv
import os
from os import path

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from pose_format import Pose
from pose_format import Pose, PoseHeader
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderDimensions

from sign_language_datasets.utils.features import PoseFeature

from ..warning import dataset_warning
from ...datasets.config import SignDatasetConfig, cloud_bucket_file

_DESCRIPTION = """
PopSign ASL v1.0 dataset collects examples of 250 isolated American Sign Language (ASL) signs using Pixel 4A smartphone selfie cameras in a variety of environments.
"""

_CITATION = """
@article{starner2024popsign,
  title={PopSign ASL v1. 0: An Isolated American Sign Language Dataset Collected via Smartphones},
  author={Starner, Thad and Forbes, Sean and So, Matthew and Martin, David and Sridhar, Rohit and Deshpande, Gururaj and Sepah, Sam and Shahryar, Sahir and Bhardwaj, Khushi and Kwok, Tyler and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
"""

_DOWNLOAD_URL = "https://signdata.cc.gatech.edu/"


# _POSE_URLS = {"holistic": cloud_bucket_file("poses/holistic/ASLCitizen.zip")} 
_POSE_HEADERS = {"holistic": path.join(path.dirname(path.realpath(__file__)), "holistic.poseheader")}


class PopSign(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Popsign dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    BUILDER_CONFIGS = [SignDatasetConfig(name="default", include_pose="holistic")]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {
            "id": tfds.features.Text(),
            "text": tfds.features.Text(),
            # "signer_id": tfds.features.Text(),
        }

        # TODO: add videos

        if self._builder_config.include_pose == "holistic":
            pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
            stride = 1 if self._builder_config.fps is None else 30 / self._builder_config.fps
            features["pose"] = PoseFeature(shape=(None, 1, 576, 3), header_path=pose_header_path, stride=stride)

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage=_DOWNLOAD_URL,
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dataset_warning(self)

        # too expensive to host the poses at the moment, need to specify a local path
        # poses_dir = str(dl_manager.download_and_extract(_POSE_URLS["holistic"]))
        poses_dir = self._builder_config.extra["pose_dir"]

        return [
            tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"poses_dir": path.join(poses_dir, 'train')}),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION, gen_kwargs={"poses_dir": path.join(poses_dir, 'val')}
            ),
            tfds.core.SplitGenerator(name=tfds.Split.TEST, gen_kwargs={"poses_dir": path.join(poses_dir, 'test')}),
        ]

    def _generate_examples(self, poses_dir: str):
        """Yields examples."""

        for label in os.listdir(poses_dir):
            subdirectory_path = os.path.join(poses_dir, label)
            if os.path.isdir(subdirectory_path):
                for filename in os.listdir(subdirectory_path):
                    if filename.endswith('.pose'):
                        datum = {"id": filename.replace('.pose', ''), "text": label}
                        mediapipe_path = path.join(subdirectory_path, filename)

                        with open(mediapipe_path, "rb") as f:
                            try:
                                pose = Pose.read(f.read())
                                datum["pose"] = pose

                                yield datum["id"], datum
                            except Exception as e:
                                print(e)

