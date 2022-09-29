"""Public DGS Corpus: parallel corpus for German Sign Language (DGS) with German and English annotations"""
from __future__ import annotations

import gzip
import json
from copy import copy

import cv2
import math

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from os import path
from typing import Dict, Any, Set, Optional, List

from pose_format.numpy import NumPyPoseBody
from pose_format.utils.openpose import load_openpose, OpenPoseFrames
from pose_format.pose import Pose

from .dgs_utils import get_elan_sentences
from ..warning import dataset_warning
from ...datasets.config import SignDatasetConfig
from ...utils.features import PoseFeature

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

_DESCRIPTION = """
Parallel corpus for German Sign Language (DGS) with German and English annotations
"""

_CITATION = """
@misc{dgscorpus_3,
  title = {MEINE DGS -- annotiert. {\"O}ffentliches Korpus der Deutschen Geb{\"a}rdensprache, 3. Release / MY DGS -- annotated. Public Corpus of German Sign Language, 3rd release},
  author = {Konrad, Reiner and Hanke, Thomas and Langer, Gabriele and Blanck, Dolly and Bleicken, Julian and Hofmann, Ilona and Jeziorski, Olga and K{\"o}nig, Lutz and K{\"o}nig, Susanne and Nishio, Rie and Regen, Anja and Salden, Uta and Wagner, Sven and Worseck, Satu and B{\"o}se, Oliver and Jahn, Elena and Schulder, Marc},
  year = {2020},
  type = {languageresource},
  version = {3.0},
  publisher = {Universit{\"a}t Hamburg},
  url = {https://doi.org/10.25592/dgs.corpus-3.0},
  doi = {10.25592/dgs.corpus-3.0}
}
"""

_HOMEPAGE = "https://www.sign-lang.uni-hamburg.de/meinedgs/"

# This `dgs.json` file was created using `create_index.py`
INDEX_URL = "https://nlp.biu.ac.il/~amit/datasets/dgs.json"

_POSE_HEADERS = {
    "holistic": path.join(path.dirname(path.realpath(__file__)), "holistic.poseheader"),
    "openpose": path.join(path.dirname(path.realpath(__file__)), "openpose.poseheader"),
}

_KNOWN_SPLITS = {
    "3.0.0-uzh-document": path.join(path.dirname(path.realpath(__file__)), "splits", "split.3.0.0-uzh-document.json"),
    "3.0.0-uzh-sentence": path.join(path.dirname(path.realpath(__file__)), "splits", "split.3.0.0-uzh-sentence.json")
}


def convert_dgs_dict_to_openpose_frames(input_dict: Dict[str, Any]) -> OpenPoseFrames:
    """
    Modifies the DGS Openpose format slightly to be compatible with the `pose_format` library. Most notably,
    changes the dict keys to integers (from strings) and adds a "frame_id" key to each frame dict.

    :param input_dict: OpenPose output, one dictionary overall, one sub-dictionary for each frame.
    :return: A dictionary of OpenPose frames that is compatible with `pose_format.utils.load_openpose`.
    """
    frames = {}  # type: OpenPoseFrames

    for frame_id_as_string, frame_dict in input_dict.items():
        frame_id = int(frame_id_as_string)
        frame_dict["frame_id"] = frame_id
        frames[frame_id] = frame_dict

    return frames


def get_openpose(openpose_path: str, fps: int, people: Optional[Set] = None, num_frames: Optional[int] = None) -> Dict[str, Pose]:
    """
    Load OpenPose in the particular format used by DGS (one single file vs. one file for each frame).

    :param openpose_path: Path to a file that contains OpenPose for all frames of a DGS video, in one single JSON.
    :param fps: Framerate.
    :param people: Specify the name prefixes of camera views that should be extracted. The default value is {"a", "b"}.
    :param num_frames: Number of frames when it is known and cannot be derived from OpenPose files. That is the case if
                       the last frame(s) of a video are missing from the OpenPose output.
    :return: Dictionary of Pose objects (one for each person) with a header specific to OpenPose and a body that
             contains a single array.
    """
    # set mutable default argument
    if people is None:
        people = {"a", "b"}

    with gzip.GzipFile(openpose_path, "r") as openpose_raw:
        openpose = json.loads(openpose_raw.read().decode("utf-8"))

    # select views if their name starts with "a" or "b"
    views = {view["camera"][0]: view for view in openpose if view["camera"][0] in people}

    poses = {p: None for p in people}
    for person, view in views.items():
        width, height, frames_obj = view["width"], view["height"], view["frames"]

        # Convert to pose format
        frames = view["frames"]
        frames = convert_dgs_dict_to_openpose_frames(frames)
        poses[person] = load_openpose(frames, fps=fps, width=width, height=height, depth=0, num_frames=num_frames)

    return poses


def load_split(split_name: str) -> Dict[str, List[str]]:
    """
    Loads a split from the file system. What is loaded must be a JSON object with the following structure:

    {"train": ..., "dev": ..., "test": ...}

    :param split_name: An identifier for a predefined split or a filepath to a custom split file.
    :return: The split loaded as a dictionary.
    """
    if split_name not in _KNOWN_SPLITS.keys():
        # assume that the supplied string is a path on the file system
        if not path.exists(split_name):
            raise ValueError(
                "Split '%s' is not a known data split identifier and does not exist as a file either.\n"
                "Known split identifiers are: %s" % (split_name, str(_KNOWN_SPLITS))
            )

        split_path = split_name
    else:
        # the supplied string is an identifier for a predefined split
        split_path = _KNOWN_SPLITS[split_name]

    with open(split_path) as infile:
        split = json.load(infile)  # type: Dict[str, List[str]]

    return split


DEFAULT_FPS = 50


class DgsCorpusConfig(SignDatasetConfig):
    def __init__(self, data_type: Literal["document", "sentence"] = "document", split: str = None, **kwargs):
        """
        :param split: An identifier for a predefined split or a filepath to a custom split file.
        :param data_type: Whether to return documents or sentences as data.
        """
        super().__init__(**kwargs)

        self.data_type = data_type
        self.split = split

        # Verify split matches data type
        if self.split in _KNOWN_SPLITS and not self.split.endswith(self.data_type):
            raise ValueError(f"Split '{self.split}' is not compatible with data type '{self.data_type}'.")


class DgsCorpus(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for dgs_corpus dataset."""

    VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES = {
        "3.0.0": "3rd release",
    }

    BUILDER_CONFIGS = [
        DgsCorpusConfig(name="default", include_video=True, include_pose="openpose"),
        DgsCorpusConfig(name="videos", include_video=True, include_pose=None),
        DgsCorpusConfig(name="openpose", include_video=False, include_pose="openpose"),
        DgsCorpusConfig(name="holistic", include_video=False, include_pose="holistic"),
        DgsCorpusConfig(name="annotations", include_video=False, include_pose=None),
        DgsCorpusConfig(name="sentences", include_video=False, include_pose=None, data_type="sentence"),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        assert isinstance(self._builder_config, DgsCorpusConfig), \
            "Builder config for dgs_corpus must be an instance of DgsCorpusConfig"

        features = {
            "id": tfds.features.Text(),
            "paths": {
                "ilex": tfds.features.Text(),
                "eaf": tfds.features.Text(),
                "srt": tfds.features.Text(),
                "cmdi": tfds.features.Text(),
            },
        }

        if self._builder_config.data_type == "sentence":
            features["document_id"] = tfds.features.Text()
            features["sentence"] = {
                "id": tfds.features.Text(),
                "participant": tfds.features.Text(),
                "start": tf.int32,
                "end": tf.int32,
                "german": tfds.features.Text(),
                "english": tfds.features.Text(),
                "glosses": tfds.features.Sequence(
                    {
                        "start": tf.int32,
                        "end": tf.int32,
                        "gloss": tfds.features.Text(),
                        "hand": tfds.features.Text(),
                        "Lexeme_Sign": tfds.features.Text(),
                        "Gebärde": tfds.features.Text(),
                        "Sign": tfds.features.Text(),
                    }
                ),
                "mouthings": tfds.features.Sequence({"start": tf.int32, "end": tf.int32, "mouthing": tfds.features.Text()}),
            }

        if self._builder_config.include_video:
            features["fps"] = tf.int32
            video_ids = ["a", "b", "c"]
            if self._builder_config.process_video:
                video_feature = self._builder_config.video_feature((640, 360))
                if self._builder_config.data_type == "document":
                    features["videos"] = {_id: video_feature for _id in video_ids}
                else:
                    features["video"] = video_feature
            features["paths"]["videos"] = {_id: tfds.features.Text() for _id in video_ids}

        # Add poses if requested
        if self._builder_config.include_pose is not None:
            pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
            stride = 1 if self._builder_config.fps is None else 50 / self._builder_config.fps

            if self._builder_config.include_pose == "openpose":
                pose_shape = (None, 1, 137, 2)
            if self._builder_config.include_pose == "holistic":
                pose_shape = (None, 1, 543, 3)

            pose_feature = PoseFeature(shape=pose_shape, stride=stride, header_path=pose_header_path)
            if self._builder_config.data_type == "document":
                features["poses"] = {_id: pose_feature for _id in ["a", "b"]}
            else:
                features["pose"] = pose_feature

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage=_HOMEPAGE,
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dataset_warning(self)

        index_path = dl_manager.download(INDEX_URL)

        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        # No need to download HTML pages
        for datum in index_data.values():
            del datum["transcript"]
            del datum["format"]

        # Don't download videos if not necessary
        if not self._builder_config.include_video:
            for datum in index_data.values():
                del datum["video_a"]
                del datum["video_b"]
                del datum["video_c"]

        # Don't download openpose poses if not necessary
        if self._builder_config.include_pose != "openpose":
            for datum in index_data.values():
                del datum["openpose"]

        # Don't download holistic poses if not necessary
        if self._builder_config.include_pose != "holistic":
            for datum in index_data.values():
                del datum["holistic_a"]
                del datum["holistic_b"]

        urls = {url: url for datum in index_data.values() for url in datum.values() if url is not None}

        local_paths = dl_manager.download(urls)

        data = {_id: {k: local_paths[v] if v is not None else None for k, v in datum.items()} for _id, datum in index_data.items()}

        if self._builder_config.split is not None:
            split = load_split(self._builder_config.split)

            train_args = {"data": data, "split": split["train"]}
            dev_args = {"data": data, "split": split["dev"]}
            test_args = {"data": data, "split": split["test"]}

            return [
                tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs=train_args),
                tfds.core.SplitGenerator(name=tfds.Split.VALIDATION, gen_kwargs=dev_args),
                tfds.core.SplitGenerator(name=tfds.Split.TEST, gen_kwargs=test_args),
            ]

        else:
            return [tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"data": data})]

    def _include_videos(self, datum: Any, features: Any):
        videos = {}
        for participant in ["a", "b", "c"]:
            video_key = "video_" + participant

            if video_key not in datum.keys() or datum[video_key] is None:
                video = ""
            else:
                video = str(datum[video_key])

            videos[participant] = video

        # make sure that the video fps is as expected
        for video_path in videos.values():
            if video_path == "":
                continue

            cap = cv2.VideoCapture(video_path)
            actual_video_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            assert math.isclose(actual_video_fps, float(DEFAULT_FPS)), "Framerate of video '%s' is %f instead of %d" % (
                video_path,
                actual_video_fps,
                DEFAULT_FPS,
            )

        features["fps"] = self._builder_config.fps if self._builder_config.fps is not None else DEFAULT_FPS
        features["paths"]["videos"] = videos

    def _generate_examples(self, data, split: List[str] | Dict[str, List[str]] = None):
        """ Yields examples. """

        default_video = np.zeros((0, 0, 0, 3))  # Empty video

        for document_id, datum in list(data.items()):
            if split is not None and document_id not in split:
                continue

            features = {
                "id": document_id,
                "paths": {t: str(datum[t]) if t in datum else "" for t in ["ilex", "eaf", "srt", "cmdi"]},
            }

            if self._builder_config.include_video:
                self._include_videos(datum, features)

            poses = None
            if self._builder_config.include_pose is not None:
                if self._builder_config.include_pose == "openpose":
                    poses = get_openpose(datum["openpose"], fps=DEFAULT_FPS)

                if self._builder_config.include_pose == "holistic":
                    poses = {}
                    for person in ["a", "b"]:
                        if datum["holistic_" + person] is not None:
                            with open(datum["holistic_" + person], "rb") as f:
                                poses[person] = Pose.read(f.read())
                        else:
                            poses[person] = None

            if self._builder_config.data_type == "document":
                if poses is not None:
                    features["poses"] = poses
                if self._builder_config.process_video:
                    features["videos"] = {t: v if v != "" else default_video for t, v in features["paths"]["videos"].items()}
                features["id"] = document_id
                yield document_id, features
            else:
                sentences = list(get_elan_sentences(datum["eaf"]))
                for sentence in sentences:
                    if split is not None and sentence["id"] not in split[document_id]:
                        continue

                    if sentence["english"] is None:
                        sentence["english"] = ""

                    features = copy(features)  # Unclear if necessary, but better safe than sorry
                    features["sentence"] = sentence

                    start_time = sentence["start"] / 1000
                    start_frame = int(start_time * DEFAULT_FPS)
                    end_time = sentence["end"] / 1000
                    end_frame = math.ceil(end_time * DEFAULT_FPS)

                    if poses is not None:
                        pose = poses[sentence["participant"].lower()]
                        sub_pose_body = NumPyPoseBody(
                            fps=pose.body.fps,
                            data=pose.body.data[start_frame:end_frame],
                            confidence=pose.body.confidence[start_frame:end_frame],
                        )
                        features["pose"] = Pose(pose.header, sub_pose_body)

                    if self._builder_config.process_video:
                        videos = features["paths"]["videos"]
                        if videos[sentence["participant"].lower()] == "":
                            features["video"] = default_video
                        else:
                            features["video"] = {
                                "video": videos[sentence["participant"].lower()],
                                "ffmpeg_args": ["-ss", str(start_time), "-to", str(end_time)],
                            }
                    document_sentence_id = f'{document_id}_{sentence["id"]}'
                    features["document_id"] = document_id
                    features["id"] = document_sentence_id
                    yield document_sentence_id, features
