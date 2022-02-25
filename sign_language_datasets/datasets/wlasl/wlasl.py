"""WLASL: A large-scale dataset for Word-Level American Sign Language"""
import json

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.io.gfile import GFile
from tqdm import tqdm

from ...datasets.config import SignDatasetConfig
from ...utils.downloaders.aslpro import download_aslpro
from ...utils.downloaders.youtube import download_youtube

_DESCRIPTION = """
A large-scale dataset for Word-Level American Sign Language
"""

_CITATION = """
@inproceedings{dataset:li2020word,
    title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
    author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
    booktitle={The IEEE Winter Conference on Applications of Computer Vision},
    pages={1459--1469},
    year={2020}
}
"""

_INDEX_URL = "https://raw.githubusercontent.com/dxli94/WLASL/0ac8108282aba99226e29c066cb8eab847ef62da/start_kit/WLASL_v0.3.json"

_POSE_URLS = {"openpose": "https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/info/wlasl.tar"}


class Wlasl(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for wlasl dataset."""

    VERSION = tfds.core.Version("0.3.0")
    RELEASE_NOTES = {"0.3.0": "fix deafasl links"}

    BUILDER_CONFIGS = [SignDatasetConfig(name="default", include_video=False, include_pose="openpose")]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {
            "id": tfds.features.Text(),
            "signer": tfds.features.Text(),
            "gloss": tfds.features.Text(),
            "gloss_variation": tf.int32,
            "video_file": {
                "start": tf.int32,
                "end": tf.int32,
                "remote": tfds.features.Text(),
            },
            "fps": tf.int32,
            "bbox": tfds.features.BBoxFeature(),
        }

        if self._builder_config.include_video:
            features["video"] = self._builder_config.video_feature((None, None))
            features["video_file"]["local"] = tfds.features.Text()

        if self._builder_config.include_pose == "openpose":
            features["pose"] = {
                "data": tfds.features.Tensor(shape=(None, 1, 137, 2), dtype=tf.float32),
                "conf": tfds.features.Tensor(shape=(None, 1, 137), dtype=tf.float32),
            }

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            supervised_keys=("video", "gloss"),
            homepage="https://dxli94.github.io/WLASL/",
            citation=_CITATION,
        )

    def _download_video(self, url: str, dl_manager):
        print("url", url)

        if "aslpro" in url:
            return dl_manager.download_custom(url, download_aslpro)
        elif "youtube" in url or "youtu.be" in url:
            return dl_manager.download_custom(url, download_youtube)
        else:
            return dl_manager.download(url)

    def _download_videos(self, data, dl_manager):
        videos = {}
        for datum in data:
            for instance in datum["instances"]:
                videos[instance["video_id"]] = instance["url"]

        paths = {}
        for video_id, video in tqdm(videos.items(), total=len(videos)):
            print(video_id, video)
            try:
                paths[video_id] = self._download_video(video, dl_manager)
            except (FileNotFoundError, ConnectionError) as err:
                print("Failed to download", video, str(err))

        return paths

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        data_index_path = dl_manager.download(_INDEX_URL)

        # Download videos and update paths
        with GFile(data_index_path, "r") as f:
            data = json.load(f)

            if self._builder_config.include_video:
                paths = self._download_videos(data, dl_manager)
                for datum in data:
                    for instance in datum["instances"]:
                        instance["video"] = paths[instance["video_id"]] if instance["video_id"] in paths else None

        if self._builder_config.include_pose == "openpose":
            pose_path = dl_manager.download_and_extract(_POSE_URLS[self._builder_config.include_pose])
        else:
            pose_path = None

        return {
            "train": self._generate_examples(data, pose_path, "train"),
            "validation": self._generate_examples(data, pose_path, "val"),
            "test": self._generate_examples(data, pose_path, "test"),
        }

    def _generate_examples(self, data, pose_path, split):
        """Yields examples."""

        counter = 0
        for datum in data:
            counter += len(datum["instances"])
        print("counter", counter)

        # print(data)
        # print(pose_path)
        # print(len(data))

        raise Exception("die")
        # counter = 0
        # for datum in data:
        #     for instance in datum["instances"]:
        #         if instance["split"] == split and instance["video"] is not None:
        #             yield counter, {
        #                 "video": instance["video"],
        #                 "url": instance["url"],
        #                 "start": instance["frame_start"],
        #                 "end": instance["frame_end"],
        #                 "fps": instance["fps"],
        #                 "signer_id": instance["signer_id"],
        #                 "bbox": instance["bbox"],
        #                 "gloss": datum["gloss"],
        #                 "gloss_variation": instance["variation_id"],
        #             }
        #         counter += 1
