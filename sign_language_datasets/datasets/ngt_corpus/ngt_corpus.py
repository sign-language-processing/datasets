"""Corpus NGT: an open access online corpus of movies with annotations of Sign Language of the Netherlands"""

import json
import re

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Optional

from ...datasets.config import SignDatasetConfig

_DESCRIPTION = """
An open access online corpus of movies with annotations of Sign Language of the Netherlands.
"""

_CITATION = """\
@inproceedings{dataset:Crasborn2008TheCN,
    title = {The Corpus NGT: An online corpus with annotations of Sign Language of the Netherlands 
    for professionals and laymen},
    author = {O. Crasborn and I. Zwitserlood},
    year = {2008}
}
"""

# This `ngt.json` file was created using `create_index.py`
INDEX_URL = "https://nlp.biu.ac.il/~amit/datasets/ngt.json"

# alternative mirror:
# INDEX_URL = "https://files.ifi.uzh.ch/cl/archiv/2022/easier/ngt.json"

_HOMEPAGE = "https://www.ru.nl/corpusngtuk/"

_VIDEO_RESOLUTION = (352, 288)

# the "combined view" of both participants is a wider video
_VIDEO_RESOLUTION_C = (704, 288)

_FRAMERATE = 25


def _get_speaker_id_from_filename(filename: str) -> Optional[str]:
    """

    :param filename:
    :return:
    """
    matches = re.findall(r"S\d+", filename)
    if len(matches) == 0:
        return None

    return matches[0]


def _get_view_to_speaker_mapping(datum_dict: dict) -> dict:
    """

    :param datum_dict:
    :return:
    """
    speaker_set = set()
    for key in datum_dict.keys():
        speaker_id = _get_speaker_id_from_filename(key)
        if speaker_id is None:
            continue
        speaker_set.add(speaker_id)
    speakers = list(sorted(speaker_set))

    assert 0 < len(speakers) <= 2, "Corpus example has 0 or more than 2 speakers: %s" % str(datum_dict)

    return {"a": speakers[0], "b": speakers[1] if len(speakers) == 2 else None}


class NGTCorpus(tfds.core.GeneratorBasedBuilder):
    """
    DatasetBuilder for NGT corpus dataset.

    Design decisions about camera views and speaker IDs:

    There are 4 different camera views:
    - "f": face of speaker, frontal view, resolution: 352 x 288
    - "b": body of speaker, including the face, frontal view, resolution: 352 x 288
    - "t": top, the speaker filmed from above, resolution: 352 x 288
    - no view identifier and no speaker (i.e. "CNGT0100.mpg" for example): wider-angle video that
      shows the entire scene, resolution: 704 x 288

    For this loader we decided to:
    - ignore the "t" (top) and "f" (face) views
    - use "a" and "b" as the video feature identifiers for the "b" (body) view of the first and second speaker
      in each corpus example, ignoring the fact that every speaker has a unique ID
    - use "c" as the video feature identifier for the combined view
    """

    VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES = {
        "3.0.0": "3rd release",
    }

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=True, include_pose=None),
        SignDatasetConfig(name="videos", include_video=True, include_pose=None),
        SignDatasetConfig(name="annotations", include_video=False, include_pose=None),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        features = {
            "id": tfds.features.Text(),
            "paths": {
                "eaf": tfds.features.Text(),
            },
        }

        if self._builder_config.include_video:
            features["fps"] = tf.int32
            video_ids = ["a", "b", "c"]
            if self._builder_config.process_video:
                features["videos"] = {"a": self._builder_config.video_feature(_VIDEO_RESOLUTION),
                                      "b": self._builder_config.video_feature(_VIDEO_RESOLUTION),
                                      "c": self._builder_config.video_feature(_VIDEO_RESOLUTION_C)}
            features["paths"]["videos"] = {_id: tfds.features.Text() for _id in video_ids}

        # Add poses if requested
        if self._builder_config.include_pose is not None:
            raise NotImplementedError("Poses are currently not available for the NGT corpus.")

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage=_HOMEPAGE,
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        index_path = dl_manager.download(INDEX_URL)

        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        # remove corpus examples if videos are restricted (= only an empty ELAN file is available, or only the "c"
        # or "t" video without annotations)
        keys_to_be_removed = set()
        for datum_key, datum in index_data.items():
            if len(datum.keys()) == 1:
                keys_to_be_removed.add(datum_key)

        # this is currently removing the following keys:
        # {'CNGT0025', 'CNGT0876', 'CNGT1542', 'CNGT0020', 'CNGT0314'}

        for key in keys_to_be_removed:
            del index_data[key]

        if self._builder_config.include_video:
            # rename "body" video keys, ignore other views
            for datum in index_data.values():

                # determine which video URL corresponds to abstract views "a" and "b"
                view_to_speaker_mapping = _get_view_to_speaker_mapping(datum)

                speaker_a = view_to_speaker_mapping["a"]
                video_key_a = "video_" + speaker_a + "_b"  # this "_b" is for "body" view
                datum["video_a"] = datum[video_key_a]

                speaker_b = view_to_speaker_mapping["b"]
                if speaker_b is not None:
                    video_key_b = "video_" + speaker_b + "_b"  # this "_b" is for "body" view
                    datum["video_b"] = datum[video_key_b]

                video_keys_to_keep = ("video_a", "video_b", "video_c")
                all_keys = list(datum.keys())

                for key in all_keys:
                    if key.startswith("video"):
                        if key not in video_keys_to_keep:
                            del datum[key]
        else:
            # Don't download videos if not necessary
            for datum in index_data.values():
                keys = list(datum.keys())
                for key in keys:
                    if key.startswith("video"):
                        del datum[key]

        # never download audio files
        for datum in index_data.values():
            if "audio" in datum.keys():
                del datum["audio"]

        urls = {url: url for datum in index_data.values() for url in datum.values() if url is not None}

        local_paths = dl_manager.download(urls)

        processed_data = {
            _id: {k: local_paths[v] if v is not None else None for k, v in datum.items()} for _id, datum in index_data.items()
        }

        return [tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"data": processed_data})]

    def _generate_examples(self, data):
        """ Yields examples. """

        default_fps = _FRAMERATE
        default_video = np.zeros((0, 0, 0, 3))  # Empty video

        for _id, datum in list(data.items()):
            features = {
                "id": _id,
                "paths": {t: str(datum[t]) if t in datum else "" for t in ["eaf"]},
            }

            if self._builder_config.include_video:
                videos = {t: str(datum["video_" + t]) if ("video_" + t) in datum else "" for t in ["a", "b", "c"]}

                features["fps"] = self._builder_config.fps if self._builder_config.fps is not None else default_fps
                features["paths"]["videos"] = videos
                if self._builder_config.process_video:
                    features["videos"] = {t: v if v != "" else default_video for t, v in videos.items()}

            if self._builder_config.include_pose == "openpose":
                raise NotImplementedError("OpenPose poses are currently not available for the NGT corpus.")

            if self._builder_config.include_pose == "holistic":
                raise NotImplementedError("Holistic poses are currently not available for the NGT corpus.")

            yield _id, features
