"""The Sem-Lex Benchmark: Modeling ASL Signs and Their Phonemes"""

import csv
from os import path

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.io.gfile import GFile

from pose_format import Pose, PoseHeader
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderDimensions

from sign_language_datasets.utils.features import PoseFeature

from ..warning import dataset_warning
from ...datasets.config import SignDatasetConfig

_DESCRIPTION = """
The Sem-Lex Benchmark contains 84k isolated American Sign Language signs from a vocabulary of 3,149.
"""

_CITATION = """
@inproceedings{kezar2023sem,
  title={The Sem-Lex Benchmark: Modeling ASL Signs and Their Phonemes},
  author={Kezar, Lee and Thomason, Jesse and Caselli, Naomi and Sehyr, Zed and Pontecorvo, Elana},
  booktitle={Proceedings of the 25th International ACM SIGACCESS Conference on Computers and Accessibility},
  pages={1--10},
  year={2023}
}
"""

_CSV_URL = "https://drive.google.com/file/d/1pkX8_TzL3kdJytQvrU68QEAp6oUvt4rv/view?usp=drive_link"
_POSE_URLS = {
    "holistic": [
        ("sem-lex-train-poses.tar.gz", "https://drive.google.com/file/d/12BlVy7S07MvF-moHoT0egdyJIJf_f7nS/view?usp=drive_link"),
        ("sem-lex-val-poses.tar.gz", "https://drive.google.com/file/d/1r7SmksY4U9GtLUR05h-fYZHZ9uup4eeI/view?usp=drive_link"),
        ("sem-lex-test-poses.tar.gz", "https://drive.google.com/file/d/1uYoM1zNpw4oLpJe4LwtDVPBNgKmAi8CC/view?usp=drive_link"),
    ]
}
_POSE_HEADERS = {"holistic": path.join(path.dirname(path.realpath(__file__)), "holistic.poseheader")}


class SemLex(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for sem-lex dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    BUILDER_CONFIGS = [SignDatasetConfig(name="default", include_pose="holistic")]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {
            "id": tfds.features.Text(),
            "text": tfds.features.Text(),
            "signer_id": tfds.features.Text(),
            "label_type": tfds.features.Text(),
            "duration": tf.float32,
        }

        # TODO: add videos

        if self._builder_config.include_pose == "holistic":
            pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
            stride = 1 if self._builder_config.fps is None else 30 / self._builder_config.fps
            features["pose"] = PoseFeature(shape=(None, 1, 553, 3), header_path=pose_header_path, stride=stride)

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="https://github.com/leekezar/SemLex",
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dataset_warning(self)

        # custom download with gdown
        csv_path = path.join(dl_manager.download_dir, "extracted", "sem-lex-metadata.csv")
        if not path.exists(csv_path):
            try:
                import gdown
            except ImportError:
                raise ImportError("Please install gdown with: pip install gdown")

            gdown.download(url=_CSV_URL, output=csv_path, quiet=False, fuzzy=True)
        self.csv_path = csv_path

        to_be_extracted = []
        
        # If it's not requested, e.g. None, don't try to download it. 
        if self._builder_config.include_pose == "holistic":
            to_be_extracted = []
            for file_name, url in _POSE_URLS[self._builder_config.include_pose]:
                output = path.join(dl_manager.download_dir, file_name)
                if not path.exists(output):
                    gdown.download(url=url, output=output, quiet=False, fuzzy=True)
                to_be_extracted.append(output)
            archives = dl_manager.extract(to_be_extracted)
        else: 
            archives = [None, None, None] # So that the indexing below will not crash, e.g. on archives[0]

        return [
            tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"archive_path": archives[0], "split": "train"}),
            tfds.core.SplitGenerator(name=tfds.Split.VALIDATION, gen_kwargs={"archive_path": archives[1], "split": "val"}),
            tfds.core.SplitGenerator(name=tfds.Split.TEST, gen_kwargs={"archive_path": archives[2], "split": "test"}),
        ]

    def _generate_examples(self, archive_path: str, split: str):
        """Yields examples."""

        with GFile(self.csv_path, "r") as csv_file:
            csv_data = csv.reader(csv_file, delimiter=",")
            next(csv_data)  # Ignore the header

            for i, row in enumerate(csv_data):
                if row[4] != split:
                    continue

                datum = {
                    "id": f"{i}_{row[0]}",  # warning: id in the original dataset is not unique
                    "text": row[6],
                    "signer_id": row[2],
                    "label_type": row[5],
                    "duration": float(row[3]),
                }

                if self.builder_config.include_pose is not None:
                    if self.builder_config.include_pose == "holistic":
                        # get data from .npy file
                        npy_path = path.join(archive_path, split, f"{row[1]}.npy")

                        if path.exists(npy_path):
                            # reorder keypoints
                            pose_data = np.load(npy_path)
                            FACE = np.arange(0, 478).tolist()
                            POSE = np.arange(478, 511).tolist()
                            LHAND = np.arange(511, 532).tolist()
                            RHAND = np.arange(532, 553).tolist()
                            points = POSE + FACE + LHAND + RHAND
                            pose_data = pose_data[:, points, :]

                            # create a new Pose instance
                            width = 640
                            height = 480
                            dimensions = PoseHeaderDimensions(width=width, height=height, depth=1000)
                            from pose_format.utils.holistic import holistic_components
                            header = PoseHeader(
                                version=0.1, dimensions=dimensions, components=holistic_components("XYZC", 10)[:-1]
                            )  # no world landmarks

                            # add the person dimension
                            pose_data = np.expand_dims(pose_data, 1)
                            # revert the scaling done in the original dataset
                            pose_data = pose_data * np.array([width, height, 1.0])
                            body = NumPyPoseBody(
                                fps=30, data=pose_data, confidence=np.ones(shape=(pose_data.shape[0], 1, header.total_points()))
                            )  # FIXME: unknown
                            pose = Pose(header, body)
                            datum["pose"] = pose
                        else:
                            datum["pose"] = None

                yield datum["id"], datum
