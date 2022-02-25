"""Public DGS Corpus: parallel corpus for German Sign Language (DGS) with German and English annotations"""

import gzip
import json

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from os import path
from typing import Dict, Any, Set, Optional
from pose_format.utils.openpose import load_openpose, OpenPoseFrames
from pose_format.pose import Pose

from ...datasets.config import SignDatasetConfig
from ...utils.features import PoseFeature

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


def get_openpose(openpose_path: str, fps: int, people: Optional[Set] = None,
                 num_frames: Optional[int] = None) -> Dict[str, Pose]:
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


class DgsCorpus(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for dgs_corpus dataset."""

    VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES = {
        "3.0.0": "3rd release",
    }

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=True, include_pose="openpose"),
        SignDatasetConfig(name="videos", include_video=True, include_pose=None),
        SignDatasetConfig(name="openpose", include_video=False, include_pose="openpose"),
        SignDatasetConfig(name="holistic", include_video=False, include_pose="holistic"),
        SignDatasetConfig(name="annotations", include_video=False, include_pose=None),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        features = {
            "id": tfds.features.Text(),
            "paths": {
                "ilex": tfds.features.Text(),
                "eaf": tfds.features.Text(),
                "srt": tfds.features.Text(),
                "cmdi": tfds.features.Text(),
            },
        }

        if self._builder_config.include_video:
            features["fps"] = tf.int32
            video_ids = ["a", "b", "c"]
            if self._builder_config.process_video:
                features["videos"] = {_id: self._builder_config.video_feature((640, 360)) for _id in video_ids}
            features["paths"]["videos"] = {_id: tfds.features.Text() for _id in video_ids}

        # Add poses if requested
        if self._builder_config.include_pose is not None:
            pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
            stride = 1 if self._builder_config.fps is None else 50 / self._builder_config.fps

            if self._builder_config.include_pose == "openpose":
                pose_shape = (None, 1, 137, 2)
            if self._builder_config.include_pose == "holistic":
                pose_shape = (None, 1, 543, 3)

            features["poses"] = {_id: PoseFeature(shape=pose_shape, stride=stride, header_path=pose_header_path) for _id in ["a", "b"]}

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

        processed_data = {
            _id: {k: local_paths[v] if v is not None else None for k, v in datum.items()} for _id, datum in index_data.items()
        }

        return [tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"data": processed_data})]

    def _generate_examples(self, data):
        """ Yields examples. """

        default_fps = 50
        default_video = np.zeros((0, 0, 0, 3))  # Empty video

        for _id, datum in list(data.items()):
            features = {
                "id": _id,
                "paths": {t: str(datum[t]) if t in datum else "" for t in ["ilex", "eaf", "srt", "cmdi"]},
            }

            if self._builder_config.include_video:
                videos = {t: str(datum["video_" + t]) if ("video_" + t) in datum else "" for t in ["a", "b", "c"]}

                features["fps"] = self._builder_config.fps if self._builder_config.fps is not None else default_fps
                features["paths"]["videos"] = videos
                if self._builder_config.process_video:
                    features["videos"] = {t: v if v != "" else default_video for t, v in videos.items()}

            if self._builder_config.include_pose == "openpose":
                features["poses"] = get_openpose(datum["openpose"], fps=default_fps)

            if self._builder_config.include_pose == "holistic":
                features["poses"] = {t: datum["holistic_" + t] for t in ["a", "b"]}

            yield _id, features
