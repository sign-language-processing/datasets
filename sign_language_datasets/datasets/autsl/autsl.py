"""AUTSL: Large Scale Signer Independent Isolated SLR Dataset (Turkish Sign Language)"""
import csv
import os
from os import path
from typing import Union
from zipfile import ZipFile

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.io.gfile import GFile
from tensorflow_datasets.core.download.resource import get_dl_dirname
from tqdm import tqdm

from ...datasets.config import SignDatasetConfig
from ...utils.features import PoseFeature

_DESCRIPTION = """
A large-scale, multimodal dataset that contains isolated Turkish sign videos.
It contains 226 signs that are performed by 43 different signers. There are 36,302 video samples in total.
It contains 20 different backgrounds with several challenges.
"""

_CITATION = """
@article{sincan2020autsl,
  title={AUTSL: A Large Scale Multi-Modal Turkish Sign Language Dataset and Baseline Methods},
  author={Sincan, Ozge Mercanoglu and Keles, Hacer Yalim},
  journal={IEEE Access},
  volume={8},
  pages={181340--181355},
  year={2020},
  publisher={IEEE}
}
"""

_OPENPOSE_DISCLAIMER = """
OpenPose skeleton for AUTSL was processed using the Single-Network Whole-Body Pose Estimation Model (BODY_135).
https://github.com/CMU-Perceptual-Computing-Lab/openpose_train/blob/master/experimental_models/README.md#single-network-whole-body-pose-estimation-model

In cases where multiple skeletons were detected in a frame, the most centered skeleton is used.
For more information, contact Necati Cihan Camgöz (cihancamgoz@gmail.com)"""

_TRAIN_VIDEOS = "http://158.109.8.102/AuTSL/data/train/train_set_vfbha39.zip"  # 18 files
_TRAIN_LABELS = "http://158.109.8.102/AuTSL/data/train/train_labels.csv"

_VALID_VIDEOS = "http://158.109.8.102/AuTSL/data/validation/val_set_bjhfy68.zip"  # 3 files
_VALID_LABELS = "https://nlp.biu.ac.il/~amit/datasets/public/autsl_validation_labels.csv"

_TEST_VIDEOS = "http://158.109.8.102/AuTSL/data/test/test_set_xsaft57.zip"  # 3 files
_TEST_LABELS = "https://nlp.biu.ac.il/~amit/datasets/public/autsl_test_labels.csv"

_POSE_URLS = {
    "holistic": "https://nlp.biu.ac.il/~amit/datasets/poses/holistic/autsl.tar.gz",
    "openpose": "https://nlp.biu.ac.il/~amit/datasets/poses/openpose/autsl.tar.gz",
}
_POSE_HEADERS = {
    "holistic": path.join(path.dirname(path.realpath(__file__)), "holistic.poseheader"),
    "openpose": path.join(path.dirname(path.realpath(__file__)), "openpose_135.poseheader"),
}


class AUTSL(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for AUTSL dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=True, include_pose=None),
        SignDatasetConfig(name="holistic", include_video=False, include_pose="holistic"),
        SignDatasetConfig(name="openpose", include_video=False, include_pose="openpose"),
    ]

    def __init__(self, **kwargs):
        super(AUTSL, self).__init__(**kwargs)

        self.train_decryption_key = "MdG3z6Eh1t"  # publically available in http://chalearnlap.cvc.uab.es/dataset/40/data/66/description/
        self.valid_decryption_key = "bhRY5B9zS2"  # publically available in http://chalearnlap.cvc.uab.es/dataset/40/data/65/description/
        self.test_decryption_key = "ds6Kvdus3o"  # publically available in http://chalearnlap.cvc.uab.es/dataset/40/data/67/description/

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {"id": tfds.features.Text(), "signer": tf.int32, "sample": tf.int32, "gloss_id": tf.int32}

        if self._builder_config.include_video:
            features["fps"] = tf.int32
            features["video"] = self._builder_config.video_feature((512, 512))
            features["depth_video"] = self._builder_config.video_feature((512, 512), 1)

        # Add poses if requested
        if self._builder_config.include_pose is not None:
            pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
            stride = 1 if self._builder_config.fps is None else 30 / self._builder_config.fps

            if self._builder_config.include_pose == "openpose":
                pose_shape = (None, 1, 135, 2)
            if self._builder_config.include_pose == "holistic":
                pose_shape = (None, 1, 543, 4)

            features["pose"] = PoseFeature(shape=pose_shape, header_path=pose_header_path, stride=stride)

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            supervised_keys=("video", "gloss_id"),
            homepage="http://chalearnlap.cvc.uab.es/dataset/40/description/",
            citation=_CITATION,
        )

    def _download_and_extract_multipart(self, dl_manager: tfds.download.DownloadManager, url: str, parts: int,
                                        pwd: str = None):
        """Download and extract multipart zip file"""

        # Write OpenPose disclaimer
        if self._builder_config.include_pose == "openpose":
            print(_OPENPOSE_DISCLAIMER)

        # Make sure not already downloaded
        dirname = get_dl_dirname(url)
        output_path = os.path.join(dl_manager._download_dir, dirname)
        output_path_extracted = os.path.join(dl_manager._extract_dir, dirname)

        print("output_path", output_path)
        print("output_path_extracted", output_path_extracted)

        if not os.path.isfile(output_path):
            parts = [url + f".{i + 1:03}" for i in range(parts)]
            files = dl_manager.download(parts)

            # Cat parts to single file
            with open(output_path, "ab") as cat_file:
                for f in files:
                    with open(f, "rb") as z:
                        cat_file.write(z.read())

        if not os.path.isdir(output_path_extracted):
            # Extract file
            os.makedirs(output_path_extracted)

            pwd_bytes = bytes(pwd, "utf-8") if pwd is not None else None

            with ZipFile(output_path, "r") as zip_obj:
                # Loop over each file
                for file in tqdm(iterable=zip_obj.namelist(), total=len(zip_obj.namelist())):
                    zip_obj.extract(member=file, path=output_path_extracted, pwd=pwd_bytes)

        return output_path_extracted

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        # Download labels
        train_labels = dl_manager.download(_TRAIN_LABELS)
        valid_labels = dl_manager.download(_VALID_LABELS)
        test_labels = dl_manager.download(_TEST_LABELS)

        # Load videos if needed
        if self._builder_config.include_video:
            train_parts = self._download_and_extract_multipart(dl_manager, url=_TRAIN_VIDEOS, parts=18,
                                                               pwd=self.train_decryption_key)
            train_videos = os.path.join(train_parts, "train")

            valid_parts = self._download_and_extract_multipart(dl_manager, url=_VALID_VIDEOS, parts=3,
                                                               pwd=self.valid_decryption_key)
            valid_videos = os.path.join(valid_parts, "val")

            test_parts = self._download_and_extract_multipart(dl_manager, url=_TEST_VIDEOS, parts=3,
                                                              pwd=self.test_decryption_key)
            test_videos = os.path.join(test_parts, "test")
        else:
            train_videos = valid_videos = test_videos = None

        # Load poses if needed

        if self._builder_config.include_pose is not None:
            pose_path = dl_manager.download_and_extract(_POSE_URLS[self._builder_config.include_pose])
            train_pose_path = path.join(pose_path, self._builder_config.include_pose, "train")
            valid_pose_path = path.join(pose_path, self._builder_config.include_pose, "validation")
            test_pose_path = path.join(pose_path, self._builder_config.include_pose, "test")
        else:
            train_pose_path = valid_pose_path = test_pose_path = None

        splits = [
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={"videos_path": test_videos, "poses_path": test_pose_path, "labels_path": test_labels},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"videos_path": train_videos, "poses_path": train_pose_path, "labels_path": train_labels},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={"videos_path": valid_videos, "poses_path": valid_pose_path, "labels_path": valid_labels},
            )
        ]

        return splits

    def _generate_examples(self, videos_path: Union[str, None], poses_path: Union[str, None],
                           labels_path: Union[str, None]):
        """Yields examples."""

        if labels_path is not None:
            with GFile(labels_path, "r") as labels_file:
                labels = {sample_id: int(label_id) for sample_id, label_id in csv.reader(labels_file)}
        else:
            labels = None

        if videos_path is not None:
            samples = {tuple(f.split("_")[:2]) for f in os.listdir(videos_path)}
        elif poses_path is not None:
            samples = {tuple(f.split(".")[0].split("_")) for f in os.listdir(poses_path)}
        elif labels_path is not None:
            samples = {tuple(k.split("_")) for k in labels.keys()}
        else:
            raise Exception("Found no samples to generate")

        for signer, sample in samples:
            datum = dict({"id": signer + "_" + sample, "signer": int(signer[6:]), "sample": int(sample[6:])})
            datum["gloss_id"] = labels[datum["id"]] if labels is not None else -1

            if videos_path is not None:
                datum["fps"] = self._builder_config.fps if self._builder_config.fps is not None else 30
                datum["video"] = os.path.join(videos_path, datum["id"] + "_color.mp4")
                datum["depth_video"] = os.path.join(videos_path, datum["id"] + "_depth.mp4")

            if poses_path is not None:
                datum["pose"] = os.path.join(poses_path, datum["id"] + ".pose")

            yield datum["id"], datum
