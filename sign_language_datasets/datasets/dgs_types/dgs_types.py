"""Public DGS Corpus: parallel corpus for German Sign Language (DGS) with German and English annotations"""

import re
from collections import defaultdict

import tensorflow as tf
import tensorflow_datasets as tfds

from os import path

from sign_language_datasets.utils.features import PoseFeature
from ..warning import dataset_warning

from ...datasets.config import SignDatasetConfig

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

_HOMEPAGE = "https://www.sign-lang.uni-hamburg.de/meinedgs/ling/types_de.html"

_POSE_HEADERS = {
    "holistic": path.join(path.dirname(path.realpath(__file__)), "holistic.poseheader"),
}

_POSE_URLS = {"holistic": "https://nlp.biu.ac.il/~amit/datasets/poses/holistic/dgs_types/"}


class DgsTypes(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for dgs_types dataset."""

    VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES = {
        "3.0.0": "3rd release",
    }

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=True, include_pose=None),
        SignDatasetConfig(name="openpose", include_video=False, include_pose="openpose"),
        SignDatasetConfig(name="holistic", include_video=False, include_pose="holistic"),
        SignDatasetConfig(name="annotations", include_video=False, include_pose=None),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        if self._builder_config.process_video:
            video_feature = {"name": tfds.features.Text(), "video": self._builder_config.video_feature((640, 360))}
        else:
            video_feature = {"name": tfds.features.Text(), "video": tfds.features.Text()}

        # Add poses if requested
        if self._builder_config.include_pose == "holistic":
            pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
            stride = 1 if self._builder_config.fps is None else 25 / self._builder_config.fps
            pose_shape = (None, 1, 543, 3)
            video_feature["pose"] = PoseFeature(shape=pose_shape, stride=stride, header_path=pose_header_path)

        features = {
            "id": tfds.features.Text(),
            "glosses": tfds.features.Sequence(tfds.features.Text()),
            "frequencies": tfds.features.Sequence(tf.int32),
            "hamnosys": tfds.features.Text(),
            "views": tfds.features.Sequence(video_feature),
        }

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage=_HOMEPAGE,
            supervised_keys=None,
            citation=_CITATION,
        )

    def get_galex_data(self, dl_manager: tfds.download.DownloadManager):
        GALEX = "https://www.sign-lang.uni-hamburg.de/galex/"
        index_urls = [f"{GALEX}tystatus/x{i}.html" for i in [2, 3, 4]]
        gloss_urls = set()
        for p in dl_manager.download(index_urls):
            with open(p, "r", encoding="utf-8") as f:
                for match in re.finditer(r"<p class=\"XREF\"><a href=\"\.\.\/(.*?)\"", f.read()):
                    match_url = match.group(1)
                    gloss_urls.add(GALEX + match_url)

        video_urls = {}
        data = []
        for p in dl_manager.download(list(gloss_urls)):
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
                gloss = re.findall(r"span class=\"Gloss\">(.*?)<", content)[0]
                video = re.findall(r"source src=\"\.\.\/(.*?)\"", content)[0]
                datum = {
                    "id": "galex_" + gloss,
                    "glosses": [gloss],
                    "frequencies": [],
                    "hamnosys": re.findall(r"a class=\"ham\".*?>(.*?)<", content)[0],
                    "views": [{"name": "front", "video": video}],
                }
                data.append(datum)
                video_urls[video] = GALEX + video

        video_paths = dl_manager.download(video_urls) if self.builder_config.include_video else video_urls
        for datum in data:
            for view in datum["views"]:
                view["video"] = video_paths[view["video"]]

        return data

    def get_dgs_data(self, dl_manager: tfds.download.DownloadManager):
        MEINE_DGS = "https://www.sign-lang.uni-hamburg.de/meinedgs/"
        dgs_index = dl_manager.download(MEINE_DGS + "ling/types_de.html")

        gloss_map = defaultdict(list)
        gloss_frequencies = defaultdict(list)

        with open(dgs_index, "r", encoding="utf-8") as f:
            for match in re.finditer(r"<p>(.*?) \((\d+) Tokens?\)( → )?(.*?)</p>", f.read()):
                gloss_id = re.findall(r"\.\.\/types\/(.*?)\.html", match.group(0))[0]
                gloss_frequency = int(match.group(2))
                gloss_frequencies[gloss_id].append(gloss_frequency)
                gloss_text = match.group(1) if match.group(3) is not None else re.findall(r">(.*?)<", match.group(1))[0]
                gloss_map[gloss_id].append(gloss_text)

        gloss_ids = list(gloss_map.keys())
        gloss_urls = {k: MEINE_DGS + "types/" + k + ".html" for k in gloss_ids}
        gloss_paths = dl_manager.download(gloss_urls)

        data = []
        video_urls = {}
        for gloss_id, glosses in gloss_map.items():
            with open(gloss_paths[gloss_id], "r", encoding="utf-8") as f:
                content = f.read()

            views = []

            video_src = re.findall(r"<source src=\"(.*?)\"", content)
            if len(video_src) > 0:
                full_src = video_src[0]
                views_info = re.findall(r"class=\"perspectives\".*?\'_(\d)\'.*?>(.*?)<", content)
                for view_id, view_name in views_info:
                    view_video_url = full_src.replace("_1.mp4", f"_{view_id}.mp4")
                    views.append({"name": view_name, "video": view_video_url})
                    video_urls[view_video_url] = view_video_url

            frequencies = gloss_frequencies[gloss_id]

            hamnosys_search = re.findall(r"class=\"hamnosys\".*?>(.*?)<", content)
            hamnosys = hamnosys_search[0] if len(hamnosys_search) > 0 else ""

            data.append({"id": gloss_id, "frequencies": frequencies, "glosses": glosses, "hamnosys": hamnosys, "views": views})

        if self.builder_config.include_video:
            video_paths = dl_manager.download(video_urls)
            for datum in data:
                for view in datum["views"]:
                    view["video"] = video_paths[view["video"]]

        return data

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dataset_warning(self)

        # Source 1: GALEX
        galex_data = self.get_galex_data(dl_manager)

        # Source 2: Meine DGS Types
        dgs_data = self.get_dgs_data(dl_manager)

        data = galex_data + dgs_data

        # Poses
        if self.builder_config.include_pose == "holistic":
            pose_urls = {datum["id"]: _POSE_URLS + f"{datum['id']}_{view['name']}.pose" for datum in data for view in datum["views"]}
            local_poses = dl_manager.download(pose_urls)

        return [tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"data": data})]

    def _generate_examples(self, data):
        """ Yields examples. """

        for datum in data:
            if not self._builder_config.process_video:
                for view in datum["views"]:
                    # convert PosixGPath to str
                    view["video"] = str(view["video"])

            yield datum["id"], datum
