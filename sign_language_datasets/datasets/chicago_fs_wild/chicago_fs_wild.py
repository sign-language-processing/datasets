"""Chicago Fingerspelling in the Wild Data Sets (ChicagoFSWild, ChicagoFSWild+)"""

import csv
import tarfile
from os import path

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.io.gfile import GFile

from pose_format import Pose

from sign_language_datasets.utils.features import PoseFeature
from ..config import cloud_bucket_file

from ..warning import dataset_warning
from ...datasets.config import SignDatasetConfig

_DESCRIPTION = """
ChicagoFSWild+ contains 55,232 fingerspelling sequences signed by 260 signers.
"""

_CITATION = """
@article{fs18iccv,
author = {B. Shi, A. Martinez Del Rio, J. Keane, D. Brentari, G. Shakhnarovich, and K. Livescu},
title = {Fingerspelling recognition in the wild with iterative visual attention},
journal = {ICCV},
year = {2019},
month = {October}
}
@article{fs18slt,
author = {B. Shi, A. Martinez Del Rio, J. Keane, J. Michaux, D. Brentari, G. Shakhnarovich, and K. Livescu},
title = {American Sign Language fingerspelling recognition in the wild},
journal = {SLT},
year = {2018},
month = {December}
}
"""

_VERSIONS = {
    "2.0.0": {"name": "ChicagoFSWildPlus", "url": "https://dl.ttic.edu/ChicagoFSWildPlus.tgz"},
    "1.0.0": {"name": "ChicagoFSWild", "url": "https://dl.ttic.edu/ChicagoFSWild.tgz"},
}

_CHICAGO_FS_WILD_PLUS_URL = "https://dl.ttic.edu/ChicagoFSWildPlus.tgz"
_CHICAGO_FS_WILD_URL = "https://dl.ttic.edu/ChicagoFSWild.tgz"

_POSE_URLS = {
    "holistic": {
        "ChicagoFSWild": cloud_bucket_file("poses/holistic/ChicagoFSWild.zip"),
        "ChicagoFSWildPlus": cloud_bucket_file("poses/holistic/ChicagoFSWildPlus.zip"),
    }
}
_POSE_HEADERS = {"holistic": path.join(path.dirname(path.realpath(__file__)), "holistic.poseheader")}


class ChicagoFSWild(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for ChicagoFSWild dataset."""

    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTES = {k: v["name"] for k, v in _VERSIONS.items()}

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=True),
        # SignDatasetConfig(name="holistic", include_video=False, include_pose='holistic'),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {
            "id": tfds.features.Text(),
            "text": tfds.features.Text(),
            "description": tfds.features.Text(),
            "video_url": tfds.features.Text(),
            "start": tfds.features.Text(),
            "signer": tfds.features.Text(),
            "metadata": {"frames": tf.int32, "width": tf.int32, "height": tf.int32},
        }

        if self._builder_config.include_video:
            if self._builder_config.process_video:
                features["video"] = self._builder_config.video_feature((480, 360))
            else:
                features["video"] = tfds.features.Sequence(tfds.features.Text())

        if self._builder_config.include_pose == "holistic":
            pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
            stride = 1 if self._builder_config.fps is None else 25 / self._builder_config.fps
            features["pose"] = PoseFeature(shape=(None, 1, 576, 3), header_path=pose_header_path, stride=stride)

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="https://home.ttic.edu/~klivescu/ChicagoFSWild.htm",
            supervised_keys=None,
            citation=_CITATION,
        )

    def _version_details(self, key: str):
        return _VERSIONS[str(self.version)][key]

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dataset_warning(self)

        archive = dl_manager.download_and_extract(self._version_details("url"))

        v_name = self._version_details("name")
        poses_dir = str(dl_manager.download_and_extract(_POSE_URLS["holistic"][v_name]))

        return [
            tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"archive_path": archive, "split": "train", "poses_dir": poses_dir}),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION, gen_kwargs={"archive_path": archive, "split": "dev", "poses_dir": poses_dir}
            ),
            tfds.core.SplitGenerator(name=tfds.Split.TEST, gen_kwargs={"archive_path": archive, "split": "test", "poses_dir": poses_dir}),
        ]

    def _generate_examples(self, archive_path: str, split: str, poses_dir: str):
        """Yields examples."""

        v_name = self._version_details("name")
        archive_directory = path.join(archive_path, v_name) if v_name == "ChicagoFSWild" else archive_path

        frames_directory = path.join(archive_directory, "frames")
        frames_directory = path.join(frames_directory, v_name, v_name) if v_name == "ChicagoFSWildPlus" else frames_directory

        if not path.exists(frames_directory):
            print("Extracting Frames Archive")
            tar = tarfile.open(path.join(archive_directory, v_name + "-Frames.tgz"))
            tar.extractall(path=frames_directory)
            tar.close()

        with GFile(path.join(archive_directory, v_name + ".csv"), "r") as csv_file:
            csv_data = csv.reader(csv_file, delimiter=",")
            next(csv_data)  # Ignore the header

            for i, row in enumerate(csv_data):
                if row[10] == split:
                    _id = row[1].replace("/", "-").replace("_(youtube)", "").replace("_(nad)", "")

                    datum = {
                        "id": _id,
                        "text": row[7],
                        "description": row[9],
                        "video_url": row[2],
                        "start": row[3],
                        "signer": row[-1],
                        "metadata": {"frames": int(row[4]), "width": int(row[5]), "height": int(row[6])},
                    }

                    if self._builder_config.include_video:
                        frames_base = path.join(frames_directory, row[1])

                        datum["video"] = [path.join(frames_base, name) for name in sorted(tf.io.gfile.listdir(frames_base))]

                    if self.builder_config.include_pose is not None:
                        if self.builder_config.include_pose == "holistic":
                            mediapipe_path = path.join(poses_dir, "pose", f"{_id}.pose")

                            if path.exists(mediapipe_path):
                                with open(mediapipe_path, "rb") as f:
                                    pose = Pose.read(f.read())
                                    datum["pose"] = pose
                            else:
                                datum["pose"] = None

                    yield _id, datum
