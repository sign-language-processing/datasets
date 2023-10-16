"""MEDIAPI-SKEL is a bilingual corpus of French Sign Language (LSF) in the form of pose data (coordinates of skeleton points) and French in the form of subtitles. It can be used for academic research."""
import os
from os import path

import tensorflow as tf
import tensorflow_datasets as tfds

from .mediapi_utils import read_mediapi_set
from ..warning import dataset_warning
from ...datasets.config import SignDatasetConfig
from ...utils.features import PoseFeature

_DESCRIPTION = """
MEDIAPI-SKEL is a bilingual corpus of French Sign Language (LSF) in the form of pose data (coordinates of skeleton points) and French in the form of subtitles. It can be used for academic research.

To constitute this corpus, we used 368 subtitled videos totaling 27 hours of LSF footage and written French produced by Média'Pi!, a media company producing bilingual content with LSF and written French .

We cannot provide the original videos, which can be accessed via a subscription to Média'Pi!. We provide the French subtitles an WebVTT format, aligned with two types of data: openpose and mediapipe holistic (in version 2 of the repository), calculated from each of the videos.

Data des cription:
X corresponds to the number of the corresponding video, from 00001 to 00368.
- Subtitles: For each X video, we provide a X.fr.vtt file containing the corresponding French subtitles.
- OpenPose: For each X video, we supply a X.zip archive containing as many json files as there are frames in the video, named X_Y_keypoints.json, where Y is the frame number. Each json file contains the coordinates of 137 points (25 for the body, 2x21 for the hands and 70 for the face).
- Mediapipe Holistic: For each X video, we supply a X.zip archive containing 5 csv files. X_N contains the image size (width and height), X_left_hand, X_right_hand, X_face and X_body contain the data for the hands, face and body respectively, with 1 line per frame. The 1st column shows the frame number, then for each point on the skeleton (21 for each hand, 468 for the face and 33 for the body), there is the following information: x (in pixels), y (in pixels), z (between -1 and 0), visibility (0 or 1 depending on whether the point is visible or not).

Directories and files:
Data directory
- openpose_zips.zip
- mediapipe_zips.zip
- subtitles.zip
Information directory
- video_information.csv: contains information about the resolution, frame rate, duration and number of frames of each video, as well as a breakdown of videos in train/dev/test sets.
- example: directory containing a short extract
MEDIAPI-SKEL is a 2D-skeleton video corpus of French Sign Language (LSF) with French subtitles. It can be used by members of public university and research structures.
"""

_CITATION = """
@inproceedings{bull-etal-2020-mediapi,
    title = "{MEDIAPI}-{SKEL} - A 2{D}-Skeleton Video Database of {F}rench {S}ign {L}anguage With Aligned {F}rench Subtitles",
    author = "Bull, Hannah  and
      Braffort, Annelies  and
      Gouiff{\`e}s, Mich{\`e}le",
    booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.743",
    pages = "6063--6068",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
"""

_POSE_HEADERS = {
    "holistic": path.join(path.dirname(path.realpath(__file__)), "holistic.header"),
    "openpose": path.join(path.dirname(path.realpath(__file__)), "openpose.header")
}

_SPLITS = {
    tfds.Split.TRAIN: "train",
    tfds.Split.VALIDATION: "dev",
    tfds.Split.TEST: "test",
}


class MediapiSkel(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mediapi_skel dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    BUILDER_CONFIGS = [SignDatasetConfig(name="default", include_pose="openpose")]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {
            "id": tfds.features.Text(),
            "metadata": {
                "fps": tf.int32,
                "height": tf.int32,
                "width": tf.int32,
                "duration": tf.float32,
                "frames": tf.int32,
            },
            "subtitles": tfds.features.Sequence({
                "start_time": tf.float32,
                "end_time": tf.float32,
                "text": tfds.features.Text(),
            }),
        }

        if self._builder_config.include_pose is not None:
            pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
            if self._builder_config.fps is not None:
                print("Pose FPS is not implemented for mediapi_skel dataset (since the original fps is not consistent)")

            if self._builder_config.include_pose == "openpose":
                pose_shape = (None, 1, 137, 2)
                raise NotImplementedError("Openpose is available, but not yet implemented for mediapi_skel dataset.")

            if self._builder_config.include_pose == "holistic":
                pose_shape = (None, 1, 75, 3)

            features["pose"] = PoseFeature(shape=pose_shape, stride=1, header_path=pose_header_path)

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="https://www.ortolang.fr/market/corpora/mediapi-skel/v2",
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dataset_warning(self)

        ZIP_PATH = os.environ.get("MEDIAPI_PATH", None)
        if ZIP_PATH is None:
            raise ValueError("MEDIAPI_PATH environment variable not set.")
        if not os.path.exists(ZIP_PATH):
            raise ValueError(f"MEDIAPI_PATH environment variable points to non-existent path: {ZIP_PATH}")

        if self._builder_config.include_video:
            raise NotImplementedError("Video does not exist for mediapi_skel dataset.")

        POSE_ZIP_PATH = os.environ.get("MEDIAPI_POSE_PATH", None)
        if POSE_ZIP_PATH is not None and not os.path.exists(POSE_ZIP_PATH):
            raise ValueError(f"MEDIAPI_POSE_PATH environment variable points to non-existent path: {POSE_ZIP_PATH}")

        return [
            tfds.core.SplitGenerator(name=name,
                                     gen_kwargs={"data": read_mediapi_set(mediapi_path=ZIP_PATH,
                                                                          pose_path=POSE_ZIP_PATH,
                                                                          split=split,
                                                                          pose_type=self._builder_config.include_pose)})
            for name, split in _SPLITS.items()
        ]

    def _generate_examples(self, data: iter):
        """ Yields examples. """

        for datum in data:
            yield datum["id"], {
                "id": datum["id"],
                "metadata": datum["metadata"],
                "subtitles": datum["subtitles"],
                "pose": datum["pose"]
            }
