"""bsl_corpus dataset."""

import os

import tensorflow_datasets as tfds
from tensorflow_datasets.core import download
from tensorflow_datasets.core import utils

from . import bsl_corpus_utils
from ...datasets.config import SignDatasetConfig

_DESCRIPTION = """
A corpus of British Sign Language.
"""

_CITATION = """\
@inproceedings{cormier:12033:sign-lang:lrec,
  author    = {Cormier, Kearsy and Fenlon, Jordan and Johnston, Trevor and Rentelis, Ramas and Schembri, Adam and Rowley, Katherine and Adam, Robert and Woll, Bencie},
  title     = {From Corpus to Lexical Database to Online Dictionary: Issues in annotation of the {BSL} Corpus and the Development of {BSL} {SignBank}},
  pages     = {7--12},
  editor    = {Crasborn, Onno and Efthimiou, Eleni and Fotinea, Stavroula-Evita and Hanke, Thomas and Kristoffersen, Jette and Mesch, Johanna},
  booktitle = {Proceedings of the {LREC2012} 5th Workshop on the Representation and Processing of Sign Languages: Interactions between Corpus and Lexicon},
  maintitle = {8th International Conference on Language Resources and Evaluation ({LREC} 2012)},
  publisher = {{European Language Resources Association (ELRA)}},
  address   = {Istanbul, Turkey},
  day       = {27},
  month     = may,
  year      = {2012},
  language  = {english},
  url       = {https://www.sign-lang.uni-hamburg.de/lrec/pub/12033.pdf}
}
"""

_HOMEPAGE = "https://bslcorpusproject.org/"

_VIDEO_RESOLUTION = (640, 360)

_FRAMERATE = 25


class BslCorpus(tfds.core.GeneratorBasedBuilder):
    """
  DatasetBuilder for BSL corpus dataset.

  The dataset currently loads annotations only, and ignores video files.
  """

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        # unknown if corpus files have release versions
        '1.0.0': 'Initial release.',
    }

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=False, include_pose=None),
        SignDatasetConfig(name="annotations", include_video=False, include_pose=None),
    ]

    def __init__(self, bslcp_username: str, bslcp_password: str, **kwargs):

        super(BslCorpus, self).__init__(**kwargs)

        self.bslcp_username = bslcp_username
        self.bslcp_password = bslcp_password

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

        return bsl_corpus_utils.DownloadManagerWithPyppeteer(
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
            username=self.bslcp_username,
            password=self.bslcp_password
        )

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        features = {
            "id": tfds.features.Text(),
            "paths": {
                "eaf": tfds.features.Text(),
            },
        }

        if self._builder_config.include_video:
            raise NotImplementedError("Videos are currently not available for the BSL corpus.")

        if self._builder_config.include_pose is not None:
            raise NotImplementedError("Poses are currently not available for the BSL corpus.")

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

        list_of_records = bsl_corpus_utils.generate_download_links(username=self.bslcp_username,
                                                                   password=self.bslcp_password,
                                                                   number_of_records=40)  # TODO: remove this limit

        index_data = {}

        for record in list_of_records:
            _id = record["record_id"]
            eaf_url = record["downloads"].get("EAF file", None)

            if eaf_url is None:
                continue
            else:
                datum = {"eaf": eaf_url}
                index_data[_id] = datum

        urls = {url: url for datum in index_data.values() for url in datum.values() if url is not None}

        local_paths = dl_manager.download(urls)

        processed_data = {
            _id: {k: local_paths[v] if v is not None else None for k, v in datum.items()} for _id, datum in index_data.items()
        }

        return [tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"data": processed_data})]

    def _generate_examples(self, data):
        """ Yields examples. """

        for _id, datum in list(data.items()):
            features = {
                "id": _id,
                "paths": {t: str(datum[t]) if t in datum else "" for t in ["eaf"]},
            }

            if self._builder_config.include_video:
                raise NotImplementedError("Videos are currently not available for the BSL corpus.")

            if self._builder_config.include_pose == "openpose":
                raise NotImplementedError("OpenPose poses are currently not available for the BSL corpus.")

            if self._builder_config.include_pose == "holistic":
                raise NotImplementedError("Holistic poses are currently not available for the BSL corpus.")

            yield _id, features
