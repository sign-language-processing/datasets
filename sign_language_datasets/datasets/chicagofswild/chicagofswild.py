"""How2Sign: A multimodal and multiview continuous American Sign Language (ASL) dataset"""
import csv
import tarfile
from os import path

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.io.gfile import GFile

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
    "2.0.0": {
        "name": "ChicagoFSWildPlus",
        "url": "https://dl.ttic.edu/ChicagoFSWildPlus.tgz"
    },
    "1.0.0": {
        "name": "ChicagoFSWild",
        "url": "https://dl.ttic.edu/ChicagoFSWild.tgz"
    }
}

_CHICAGO_FS_WILD_PLUS_URL = 'https://dl.ttic.edu/ChicagoFSWildPlus.tgz'
_CHICAGO_FS_WILD_URL = 'https://dl.ttic.edu/ChicagoFSWild.tgz'


class ChicagoFSWild(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for how2sign dataset."""

    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTES = {k: v["name"] for k, v in _VERSIONS.items()}

    BUILDER_CONFIGS = [SignDatasetConfig(name="default", include_video=True)]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {
            "id": tfds.features.Text(),
            "text": tfds.features.Text(),
            "description": tfds.features.Text(),
            "video_url": tfds.features.Text(),
            "start": tfds.features.Text(),
            "signer": tfds.features.Text(),
            "metadata": {
                "frames": tf.int32,
                "width": tf.int32,
                "height": tf.int32,
            },
        }

        if self._builder_config.include_video:
            features["video"] = self._builder_config.video_feature((480, 360))

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

        archive = dl_manager.download_and_extract(self._version_details("url"))

        return [
            tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"archive_path": archive, "split": "train"}),
            tfds.core.SplitGenerator(name=tfds.Split.VALIDATION, gen_kwargs={"archive_path": archive, "split": "dev"}),
            tfds.core.SplitGenerator(name=tfds.Split.TEST, gen_kwargs={"archive_path": archive, "split": "test"}),
        ]

    def _generate_examples(self, archive_path: str, split: str):
        """ Yields examples. """

        v_name = self._version_details("name")
        archive_directory = path.join(archive_path, v_name) if v_name == "ChicagoFSWild" else archive_path

        frames_directory = path.join(archive_directory, "frames")
        frames_directory = path.join(frames_directory, v_name) if v_name == "ChicagoFSWildPlus" else frames_directory

        if not path.exists(frames_directory):
            print("Extracting Frames Archive")
            tar = tarfile.open(path.join(archive_directory, v_name + "-Frames.tgz"))
            tar.extractall(path=frames_directory)
            tar.close()

        with GFile(path.join(archive_directory, v_name + ".csv"), "r") as csv_file:
            csv_data = csv.reader(csv_file, delimiter=',')
            next(csv_data)  # Ignore the header

            for row in csv_data:
                if row[10] == split:
                    _id = row[1].replace("/", "-").replace("_(youtube)", "").replace("_(nad)", "")

                    datum = {
                        "id": _id,
                        "text": row[7],
                        "description": row[9],
                        "video_url": row[2],
                        "start": row[3],
                        "signer": row[-1],
                        "metadata": {
                            "frames": int(row[4]),
                            "width": int(row[5]),
                            "height": int(row[6])
                        }
                    }

                    if self._builder_config.include_video:
                        frames_base = path.join(frames_directory, row[1])

                        datum["video"] = [path.join(frames_base, name) for name in
                                          sorted(tf.io.gfile.listdir(frames_base))]

                    yield _id, datum
