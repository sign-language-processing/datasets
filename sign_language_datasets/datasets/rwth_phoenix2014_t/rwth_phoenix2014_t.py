"""RWTH-PHOENIX 2014 T: Parallel Corpus of Sign Language Video, Gloss and Translation"""
import csv
from os import path

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.io.gfile import GFile

from ...datasets.config import SignDatasetConfig
from ...utils.features import PoseFeature

_DESCRIPTION = """
Parallel Corpus of German Sign Language of the weather, including video, gloss and translation.
Additional poses extracted by MediaPipe Holistic are also available.
"""

_CITATION = """
@inproceedings{cihan2018neural,
  title={Neural sign language translation},
  author={Cihan Camgoz, Necati and Hadfield, Simon and Koller, Oscar and Ney, Hermann and Bowden, Richard},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7784--7793},
  year={2018}
}
@article{koller2015continuous,
  title={Continuous sign language recognition: 
  Towards large vocabulary statistical recognition systems handling multiple signers},
  author={Koller, Oscar and Forster, Jens and Ney, Hermann},
  journal={Computer Vision and Image Understanding},
  volume={141},
  pages={108--125},
  year={2015},
  publisher={Elsevier}
}
"""

_VIDEO_ANNOTATIONS_URL = "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz"
_ANNOTATIONS_URL = "https://nlp.biu.ac.il/~amit/datasets/public/phoenix-annotations.tar.gz"

_POSE_URLS = {"holistic": "https://nlp.biu.ac.il/~amit/datasets/poses/holistic/phoenix.tar.gz"}
_POSE_HEADERS = {"holistic": path.join(path.dirname(path.realpath(__file__)), "pose.header")}


class RWTHPhoenix2014T(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for rwth_phoenix_2014_t dataset."""

    VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES = {"3.0.0": "Initial release."}

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=True, include_pose="holistic"),
        SignDatasetConfig(name="videos", include_video=True, include_pose=None),
        SignDatasetConfig(name="poses", include_video=False, include_pose="holistic"),
        SignDatasetConfig(name="annotations", include_video=False, include_pose=None),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {
            "id": tfds.features.Text(),
            "signer": tfds.features.Text(),
            "gloss": tfds.features.Text(),
            "text": tfds.features.Text(),
        }

        if self._builder_config.include_video:
            features["fps"] = tf.int32
            features["video"] = self._builder_config.video_feature((210, 260))

        if self._builder_config.include_pose == "holistic":
            pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
            stride = 1 if self._builder_config.fps is None else 25 / self._builder_config.fps
            features["pose"] = PoseFeature(shape=(None, 1, 543, 3), header_path=pose_header_path, stride=stride)

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/",
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        urls = [_VIDEO_ANNOTATIONS_URL if self._builder_config.include_video else _ANNOTATIONS_URL]

        if self._builder_config.include_pose is not None:
            urls.append(_POSE_URLS[self._builder_config.include_pose])

        downloads = dl_manager.download_and_extract(urls)
        annotations_path = path.join(downloads[0], "PHOENIX-2014-T-release-v3", "PHOENIX-2014-T")

        if self._builder_config.include_pose == "holistic":
            pose_path = path.join(downloads[1], "holistic")
        else:
            pose_path = None

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={"annotations_path": annotations_path, "pose_path": pose_path, "split": "dev"},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={"annotations_path": annotations_path, "pose_path": pose_path, "split": "test"},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"annotations_path": annotations_path, "pose_path": pose_path, "split": "train"},
            ),
        ]

    def _generate_examples(self, annotations_path: str, pose_path: str, split: str):
        """ Yields examples. """

        filepath = path.join(annotations_path, "annotations", "manual", "PHOENIX-2014-T." + split + ".corpus.csv")
        images_path = path.join(annotations_path, "features", "fullFrame-210x260px", split)
        poses_path = path.join(pose_path, split) if pose_path is not None else None

        with GFile(filepath, "r") as f:
            data = csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE)
            for row in data:
                datum = {
                    "id": row["name"],
                    "signer": row["speaker"],
                    "gloss": row["orth"],
                    "text": row["translation"],
                }

                if self._builder_config.include_video:
                    frames_base = path.join(images_path, row["video"])[:-7]
                    datum["video"] = [
                        path.join(frames_base, name)
                        for name in sorted(tf.io.gfile.listdir(frames_base))
                        if name != "createDnnTrainingLabels-profile.py.lprof"
                    ]
                    datum["fps"] = self._builder_config.fps if self._builder_config.fps is not None else 25

                if poses_path is not None:
                    datum["pose"] = path.join(poses_path, datum["id"] + ".pose")

                yield datum["id"], datum
