"""bobsl dataset."""
import os
import json
import requests

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import download
from tensorflow_datasets.core import utils

from pose_format.utils.openpose import load_openpose_directory

from ...datasets.config import SignDatasetConfig
from ...utils.features import PoseFeature
from ...utils.downloaders import download_auth

_DESCRIPTION = """BOBSL is a large-scale dataset of British Sign Language (BSL)."""

_CITATION = """@InProceedings{Albanie2021bobsl,
    author       = "Samuel Albanie and G{\"u}l Varol and Liliane Momeni and Hannah Bull and Triantafyllos Afouras 
    and Himel Chowdhury and Neil Fox and Bencie Woll and Rob Cooper and Andrew McParland and Andrew Zisserman",
    title        = "{BOBSL}: {BBC}-{O}xford {B}ritish {S}ign {L}anguage {D}ataset",
    howpublished = "\\url{https://www.robots.ox.ac.uk/~vgg/data/bobsl}",
    year         = "2021",
}
"""

_HOMEPAGE = "https://www.robots.ox.ac.uk/~vgg/data/bobsl/"

INDEX_URL = "https://files.ifi.uzh.ch/cl/archiv/2022/easier/bobsl.json"

_FRAMERATE = 25

_VIDEO_RESOLUTION = (444, 444)  # (width, height)

_OPENPOSE_HEADER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "openpose.poseheader")


def _add_subtitles_to_index(index_data: dict, filename: str, folder: str, subtitle_alignment_method: str) -> dict:
    """

    :param index_data:
    :param filename:
    :param folder:
    :param subtitle_alignment_method:
    :return:
    """
    assert ".vtt" in filename
    example_id = filename.replace(".vtt", "")
    filepath = os.path.join(folder, filename)

    if example_id not in index_data.keys():
        index_data[example_id] = {}

    index_data[example_id]["subtitles"] = filepath
    index_data[example_id]["subtitle_alignment_method"] = subtitle_alignment_method

    return index_data


def _walk_subtitles_and_add_to_index(index_data: dict, extracted_path: str) -> dict:
    """

    :param index_data:
    :param extracted_path: A local path returned by a tfds download manager.
    :return:
    """
    manually_aligned_folder = os.path.join(extracted_path, "subtitles", "manually-aligned")
    audio_aligned_folder = os.path.join(extracted_path, "subtitles", "audio-aligned")

    for filename in os.listdir(manually_aligned_folder):
        index_data = _add_subtitles_to_index(index_data=index_data, filename=filename, folder=manually_aligned_folder,
                                             subtitle_alignment_method="manual")

    for filename in os.listdir(audio_aligned_folder):
        index_data = _add_subtitles_to_index(index_data=index_data, filename=filename, folder=audio_aligned_folder,
                                             subtitle_alignment_method="audio")

    return index_data


def _add_spottings_to_index(index_data: dict, extracted_path: str) -> dict:
    """
    Structure of folder:

    spottings
    |- attention_spottings.json
    |- dict_spottings.json
    |- mouthings.json

    Each JSON file has the following structure:
    - Outermost keys, json_dict.keys(): ['train', 'public_test', 'val']
    - First 5 keys of json_dict["train"]: ['aachen', 'aardvark', 'aaron', 'ab', 'aba']
    - Keys of json_dict["train"]["aachen"]: ['global_times', 'names', 'probs']

    `global_times` corresponds to times in seconds in a video. `names` has dataset example ids. `probs` has
    one probability for each spotting.

    :param index_data:
    :param extracted_path:
    :return:
    """
    paths = {"spottings_attention": os.path.join(extracted_path, "spottings", "attention_spottings.json"),
             "spottings_dict": os.path.join(extracted_path, "spottings", "dict_spottings.json"),
             "spottings_mouthings": os.path.join(extracted_path, "spottings", "mouthings.json")}

    for spottings_type, spottings_path in paths.items():

        with open(spottings_path, "r") as infile:
            json_dict = json.load(infile)

            for split, gloss_dict in json_dict.items():
                for gloss_key, gloss_value_dict in gloss_dict.items():
                    global_times = gloss_value_dict["global_times"]
                    names = gloss_value_dict["names"]
                    probs = gloss_value_dict["probs"]

                    for global_time, name, prob in zip(global_times, names, probs):
                        assert name in index_data.keys()

                        if spottings_type not in index_data[name].keys():
                            index_data[name][spottings_type] = []
                        index_data[name][spottings_type].append({"global_time": global_time,
                                                                 "prob": prob,
                                                                 "gloss": gloss_key})

    return index_data


def _download_and_maybe_extract(index_data: dict,
                                dl_manager: download_auth.DownloadManagerWithAuth) -> dict:
    """

    :param index_data:
    :param dl_manager:
    :return:
    """
    urls_to_download = {}
    urls_to_download_and_extract = {}

    for datum in index_data.values():
        for url in datum.values():
            if url.endswith(".tar.gz"):
                urls_to_download_and_extract[url] = url
            else:
                urls_to_download[url] = url

    if urls_to_download:
        local_paths_downloaded = dl_manager.download(urls_to_download)
    else:
        local_paths_downloaded = {}

    if urls_to_download_and_extract:
        local_paths_downloaded_and_extracted = dl_manager.download_and_extract(urls_to_download_and_extract)
    else:
        local_paths_downloaded_and_extracted = {}

    local_paths = {**local_paths_downloaded, **local_paths_downloaded_and_extracted}

    return local_paths


class Bobsl(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for bobsl dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=True, include_pose="openpose"),
        SignDatasetConfig(name="annotations", include_video=False, include_pose=None),
        SignDatasetConfig(name="videos", include_video=True, include_pose=None),
        SignDatasetConfig(name="openpose", include_video=False, include_pose="openpose"),
    ]

    def __init__(self, bobsl_username: str, bobsl_password: str, **kwargs):

        super(Bobsl, self).__init__(**kwargs)

        self.bobsl_username = bobsl_username
        self.bobsl_password = bobsl_password

    def _make_download_manager(self, download_dir, download_config):
        """Creates a new download manager object."""
        download_dir = (
                download_dir or os.path.join(self._data_dir_root, "downloads"))
        extract_dir = (
                download_config.extract_dir or os.path.join(download_dir, "extracted"))
        manual_dir = (
                download_config.manual_dir or os.path.join(download_dir, "manual"))

        if download_config.register_checksums:
            # Note: Error will be raised here if user try to record checksums
            # from a `zipapp`
            # noinspection PyTypeChecker
            register_checksums_path = utils.to_write_path(self._checksums_path)
        else:
            register_checksums_path = None

        return download_auth.DownloadManagerWithAuth(
            download_dir=download_dir,
            extract_dir=extract_dir,
            manual_dir=manual_dir,
            url_infos=self.url_infos,
            manual_dir_instructions=self.MANUAL_DOWNLOAD_INSTRUCTIONS,
            force_download=(download_config.download_mode == download.GenerateMode.FORCE_REDOWNLOAD),
            force_extraction=(download_config.download_mode == download.GenerateMode.FORCE_REDOWNLOAD),
            force_checksums_validation=download_config.force_checksums_validation,
            register_checksums=download_config.register_checksums,
            register_checksums_path=register_checksums_path,
            verify_ssl=download_config.verify_ssl,
            dataset_name=self.name,
            username=self.bobsl_username,
            password=self.bobsl_password
        )

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        spottings_feature_dict = {"global_time": tf.float32,
                                  "prob": tf.float32,
                                  "gloss": tfds.features.Text()}

        spottings_feature_sequence = tfds.features.Sequence(spottings_feature_dict, length=None)

        features = {
            "id": tfds.features.Text(),
            "paths": {
                "subtitles": tfds.features.Text(),
            },
            "subtitle_alignment_method": tfds.features.Text(),
            "spottings": {"spottings_attention": spottings_feature_sequence,
                          "spottings_dict": spottings_feature_sequence,
                          "spottings_mouthings": spottings_feature_sequence}
        }

        # add video features if requested
        if self._builder_config.include_video:
            features["fps"] = tf.int32
            features["paths"]["video"] = tfds.features.Text()

            if self._builder_config.process_video:
                features["video"] = self._builder_config.video_feature(_VIDEO_RESOLUTION)

        # add pose features if requested
        if self._builder_config.include_pose == "holistic":
            raise NotImplementedError("Holistic poses are currently not available for the BOBSL corpus.")
        elif self._builder_config.include_pose == "openpose":
            stride = 1 if self._builder_config.fps is None else _FRAMERATE / self._builder_config.fps
            pose_shape = (None, 1, 137, 2)

            features["poses"] = PoseFeature(shape=pose_shape, stride=stride, header_path=_OPENPOSE_HEADER)

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage=_HOMEPAGE,
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: download_auth.DownloadManagerWithAuth):
        """Returns SplitGenerators."""

        # download index without dl_manager, since it would try to authenticate
        index_content = requests.get(INDEX_URL).content.decode("utf-8")
        index_data = json.loads(index_content)

        # save tar.gz urls, then delete from index
        subtitles_url = index_data["subtitles"]
        spottings_url = index_data["spottings"]

        del index_data["subtitles"]
        del index_data["spottings"]

        # Don't download videos if not necessary
        if not self._builder_config.include_video:
            for datum in index_data.values():
                del datum["video"]

        # Never download flows at the moment
        for datum in index_data.values():
            del datum["flow"]

        # Don't download poses if not necessary
        if self._builder_config.include_pose != "openpose":
            for datum in index_data.values():
                del datum["openpose"]

        # download or download-and-extract, depending on file type
        local_paths = _download_and_maybe_extract(index_data=index_data, dl_manager=dl_manager)

        processed_data = {}

        for _id, datum in index_data.items():
            processed_data[_id] = {}
            for key, url in datum.items():
                processed_data[_id][key] = local_paths[url]

        # download and extract subtitles, add to local paths
        subtitles_extracted_path = dl_manager.download_and_extract(subtitles_url)

        processed_data = _walk_subtitles_and_add_to_index(index_data=processed_data,
                                                          extracted_path=subtitles_extracted_path)

        # download and extract spottings, add to local paths
        spottings_extracted_path = dl_manager.download_and_extract(spottings_url)

        processed_data = _add_spottings_to_index(index_data=processed_data,
                                                 extracted_path=spottings_extracted_path)

        one_example = {"5085344787448740525": processed_data["5085344787448740525"]}

        with open("processed_data.json", "w") as outfile:
            json.dump(one_example, outfile)

        return [tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"data": processed_data})]

    def _generate_examples(self, data):
        """ Yields examples. """

        def _return_dict_value_or_empty_list(datum: dict, dict_key: str) -> list:
            """

            :param datum:
            :param dict_key:
            :return:
            """
            if dict_key in datum.keys():
                return datum[dict_key]
            else:
                return []

        for _id, datum in list(data.items()):
            features = {
                "id": _id,
                "paths": {"subtitles": str(datum["subtitles"])},
                "subtitle_alignment_method": datum["subtitle_alignment_method"],
                "spottings": {"spottings_attention": _return_dict_value_or_empty_list(datum, "spottings_attention"),
                              "spottings_dict": _return_dict_value_or_empty_list(datum, "spottings_dict"),
                              "spottings_mouthings": _return_dict_value_or_empty_list(datum, "spottings_mouthings")}
            }

            if self._builder_config.include_video:

                features["fps"] = self._builder_config.fps if self._builder_config.fps is not None else _FRAMERATE
                features["paths"]["video"] = datum["video"]
                if self._builder_config.process_video:
                    features["video"] = datum["video"]

            if self._builder_config.include_pose == "openpose":
                features["poses"] = load_openpose_directory(directory=datum["openpose"],
                                                            fps=_FRAMERATE,
                                                            width=_VIDEO_RESOLUTION[0],
                                                            height=_VIDEO_RESOLUTION[1],
                                                            depth=0,
                                                            num_frames=None)

            if self._builder_config.include_pose == "holistic":
                raise NotImplementedError("Holistic poses are currently not available for the BOBSL corpus.")

            yield _id, features
