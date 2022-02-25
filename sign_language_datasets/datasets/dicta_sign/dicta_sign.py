"""A list of 1000+ concepts with a sign language equivalent in each of the four project languages."""
from os import path
import tensorflow_datasets as tfds
import re

from ...datasets import SignDatasetConfig
from ...utils.features import PoseFeature

_DESCRIPTION = """
A list of 1000+ concepts with a sign language equivalent in each of the four project languages
"""

_CITATION = """
@inproceedings{efthimiou2010dicta,
  title={Dicta-sign--sign language recognition, generation and modelling: a research effort with applications in deaf communication},
  author={Efthimiou, Eleni and Fontinea, Stavroula-Evita and Hanke, Thomas and Glauert, John and Bowden, Rihard and Braffort, Annelies and Collet, Christophe and Maragos, Petros and Goudenove, Fran{\c{c}}ois},
  booktitle={Proceedings of the 4th Workshop on the Representation and Processing of Sign Languages: Corpora and Sign Language Technologies},
  pages={80--83},
  year={2010}
}
"""

SPOKEN_LANGUAGES = {
    "BSL": "en",
    "DGS": "de",
    "LSF": "fr",
    "GSL": "el"
}

_POSE_URLS = {"holistic": "https://nlp.biu.ac.il/~amit/datasets/poses/holistic/dicta_sign.tar.gz"}
_POSE_HEADERS = {"holistic": path.join(path.dirname(path.realpath(__file__)), "holistic.header")}


class DictaSign(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for sign2mint dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=True, include_pose="holistic"),
        SignDatasetConfig(name="poses", include_video=False, include_pose="holistic"),
        SignDatasetConfig(name="annotations", include_video=False),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {
            "id": tfds.features.Text(),
            "signed_language": tfds.features.Text(),
            "spoken_language": tfds.features.Text(),
            "text": tfds.features.Text(),
            "gloss": tfds.features.Text(),
            "hamnosys": tfds.features.Text()
        }

        resolution = (320, 240)
        if self._builder_config.resolution is None:
            # Required in order to pad all videos equally
            self._builder_config.resolution = resolution

        if self._builder_config.include_video and self._builder_config.process_video:
            features["video"] = self._builder_config.video_feature(resolution)
        else:
            features["video"] = tfds.features.Text()

        if self._builder_config.include_pose == "holistic":
            pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
            stride = 1 if self._builder_config.fps is None else 25 / self._builder_config.fps
            features["pose"] = PoseFeature(shape=(None, 1, 543, 3), header_path=pose_header_path, stride=stride)

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/",
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        concepts_path = dl_manager.download(
            "https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/concepts_eng.html")

        if self._builder_config.include_pose is not None:
            poses_path = dl_manager.download_and_extract(_POSE_URLS[self._builder_config.include_pose])
        else:
            poses_path = None

        regex = r"<a href=\"cs\/cs_(\d*)\.html\"><img"
        with open(concepts_path, "r", encoding="utf-8") as concepts_f:
            matches = re.finditer(regex, concepts_f.read(), re.MULTILINE)
            concept_url = "https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/cs/cs_{}.html"
            concept_urls = [concept_url.format(match.groups()[0]) for match in matches]

        concept_paths = dl_manager.download(concept_urls)

        # These video suffixes are not found
        filter_videos = ["na.webm", "bsl/293.webm", "bsl/596.webm", "gsl/998.webm", "gsl/999.webm"]

        regex = r"\"title-c\">(.*?)<[\s\S]*?title-l1\">(.*?)&[\s\S]*?<source src=\".\/..\/(.*?)\" type=\"video\/webm[\s\S]*?gloss\">(.*?)<\/div[\s\S]*?\"hns\">(.*?)<"
        concepts = []
        for concept_path in concept_paths:
            with open(concept_path, "r", encoding="utf-8") as concept_f:
                matches = re.findall(regex, concept_f.read(), re.MULTILINE)

                assert len(matches) == 4, "Concept does not include 4 matches {}".format(concept_path)

                for match in matches:
                    concept = {
                        "id": match[2].split("/")[1].split(".")[0] + "_" + match[1],
                        "signed_language": match[1],
                        "spoken_language": SPOKEN_LANGUAGES[match[1]],
                        "text": match[0],
                        "gloss": match[3],
                        "hamnosys": match[4],
                        "video": "https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/" + match[2]
                    }
                    if not any(map(lambda t: concept["video"].endswith(t), filter_videos)):
                        concepts.append(concept)

        if self._builder_config.include_video:
            video_urls = [c["video"] for c in concepts]
            video_paths = dl_manager.download(video_urls)

            for video_path, concept in zip(video_paths, concepts):
                concept["video"] = video_path if self._builder_config.process_video else str(video_path)

        return {
            'train': self._generate_examples(concepts, poses_path)
        }

    def _generate_examples(self, concepts, poses_path: str):
        """Yields examples."""

        for concept in concepts:
            if poses_path is not None:
                concept["pose"] = path.join(poses_path, 'dicta_sign', concept["id"] + ".pose")

            yield concept["id"], concept
