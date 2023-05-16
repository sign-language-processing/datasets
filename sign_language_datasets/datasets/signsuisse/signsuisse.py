"""A Swiss Sign Language Lexicon, combines all three Swiss sign languages: the German-Swiss Sign Language (DSGS), the Langue des Signes Française (LSF) and the Lingua Italiana dei Segni (LIS). ."""
import json
import re
import string

import hashlib
from os import path

import tensorflow_datasets as tfds
from pose_format import Pose

from sign_language_datasets.utils.features import PoseFeature
from tqdm import tqdm

from ..warning import dataset_warning
from ...datasets import SignDatasetConfig

_DESCRIPTION = """
Ein umfangreiches interaktives Lexikon der drei Gebärdensprachen der Schweiz (Deutschschweizerische Gebärdensprache DSGS, Langue des Signes Française LSF und Lingua Italiana dei Segni LIS). Herausgegeben vom Schweizerischen Gehörlosenbund (SGB-FSS)."""

_CITATION = """
@misc{signsuisse,
    title = {{Gehörlosenbund Gebärdensprache-Lexikon}},
    author = {{Schweizerischer Gehörlosenbund SGB-FSS}},
    year = {2023},
    howpublished = {\url{https://signsuisse.sgb-fss.ch/}},
    note = {Accessed on: \today}
}
"""

SITE_URL = "https://signsuisse.sgb-fss.ch"

_POSE_URLS = {"holistic": "https://nlp.biu.ac.il/~amit/datasets/poses/holistic/signsuisse.tar.gz"}
_POSE_HEADERS = {"holistic": path.join(path.dirname(path.realpath(__file__)), "holistic.poseheader")}


class SignSuisse(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for signsuisse dataset."""

    MAX_SIMULTANEOUS_DOWNLOADS = 1  # SignSuisse allows for a maximum of 1 simultaneous connection
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial crawl.",
    }

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=True),
        SignDatasetConfig(name="annotations", include_video=False),
        SignDatasetConfig(name="holistic", include_video=False, include_pose='holistic'),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Return s the dataset metadata."""

        features = {
            "id": tfds.features.Text(),
            "name": tfds.features.Text(),
            "category": tfds.features.Text(),
            "spokenLanguage": tfds.features.Text(),
            "signedLanguage": tfds.features.Text(),
            "url": tfds.features.Text(),
            "paraphrase": tfds.features.Text(),
            "definition": tfds.features.Text(),
            "exampleText": tfds.features.Text(),
        }

        if self._builder_config.include_video and self._builder_config.process_video:
            features["video"] = self._builder_config.video_feature((640, 480))
            features["exampleVideo"] = self._builder_config.video_feature((640, 480))
        else:
            features["video"] = tfds.features.Text()
            features["exampleVideo"] = tfds.features.Text()

        if self._builder_config.include_pose == "holistic":
            pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
            stride = 1 if self._builder_config.fps is None else 25 / self._builder_config.fps
            features["pose"] = PoseFeature(shape=(None, 1, 543, 3),
                                           header_path=pose_header_path,
                                           stride=stride,
                                           include_path=True)
            features["examplePose"] = features["pose"]

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage=SITE_URL,
            supervised_keys=None,
            citation=_CITATION,
        )

    def _list_all_lexicon_items(self, dl_manager: tfds.download.DownloadManager):
        languages = ['de', 'fr', 'it']
        dl_links = [f"https://signsuisse.sgb-fss.ch/index.php?eID=sitemap&lang={l}&format=json"
                    for l in languages]
        # FYI: to reset the indexes, run `rm $(ls *sitemap*)` in the `tensorflow_datasets/downloads/` folder

        indexes = dl_manager.download(dl_links)
        results = []
        for index in indexes:
            with open(index, "r", encoding="utf-8") as f:
                index = json.load(f)
            results.extend(index)

        if self._builder_config.sample_size is not None:
            return results[:self._builder_config.sample_size]
        return results

    def _parse_item(self, item, item_page):
        item["name"] = item["name"].replace("   ", " ").replace("  ", " ")
        with open(item_page, "r", encoding="utf-8") as f:
            html = f.read()

        # Verify that the page matches the item
        title = re.search(r"<h1.*?>(.*?)</h1>", html).group(1).strip()
        title = title.replace("&amp;", "&")
        if title != item["name"]:
            raise ValueError("Title does not match item name")

        assert title == item["name"]

        example = re.search(r"Beispiel</h2> <p>(.*?)</p>.*?<video.*?src=\"(.*?)\"", html)
        if example is not None:
            example_text = example.group(1).strip()
            example_video = SITE_URL + example.group(2).strip()
        else:
            # This happens on: https://signsuisse.sgb-fss.ch/de/lexikon/g/shakespeare-william/
            # Because the page is formatted differently. I requested a change to the page.
            example_text = ""
            example_video = ""

        video = SITE_URL + re.search(r"<video id=\"video-main\".*?src=\"(.*?)\"", html).group(1).strip()
        paraphrase_match = re.search(r"Umschreibung</h2> <p>(.*?)</p>", html)
        paraphrase = paraphrase_match.group(1).strip() if paraphrase_match else ""
        definition_match = re.search(r"Definition</h2> <p>(.*?)</p>", html)
        definition = definition_match.group(1).strip() if definition_match else ""
        category_match = re.search(r"<strong>Kategorien:<\/strong>[\s\S]*?<span>([\s\S]*?)<\/span", html)
        category = category_match.group(1).strip() if category_match else item["kategorie"]

        return {
            "id": item["uid"],
            "name": item["name"],
            "category": category,
            "spokenLanguage": item["sprache"],
            "signedLanguage": "ch-" + item["sprache"],
            "url": item["link"],
            "paraphrase": paraphrase,
            "definition": definition,
            "exampleText": example_text,
            "exampleVideo": example_video,
            "video": video
        }

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        dataset_warning(self)
        print(
            "The lexicon is available free of charge, so we look forward to your donation! https://www.sgb-fss.ch/spenden/jetzt-spenden/")

        lexicon_items = self._list_all_lexicon_items(dl_manager)
        print("Found", len(lexicon_items), "lexicon items.")
        item_urls = [item["link"] for item in lexicon_items]
        items_pages = dl_manager.download(item_urls)

        data = []
        for item, item_page in zip(lexicon_items, items_pages):
            try:
                item = self._parse_item(item, item_page)
                if item is not None:
                    data.append(item)
            except Exception as e:
                print("Failed to parse item")
                print(item)
                print(item_page)
                print(e)
                print("\n\n\n\n\n\n")
                # raise e

        # Download videos if requested
        if self._builder_config.include_video:
            video_urls = [item["video"] for item in data]
            videos = dl_manager.download(video_urls)
            for datum, video in zip(data, videos):
                datum["video"] = video  # PosixPath
                if not self._builder_config.process_video:
                    datum["video"] = str(datum["video"])

            # Download example videos if requested
            data_with_examples = [item for item in data if item["exampleVideo"] != ""]
            video_urls = [item["exampleVideo"] for item in data_with_examples]
            videos = dl_manager.download(video_urls)
            for datum, video in zip(data_with_examples, videos):
                datum["exampleVideo"] = video  # PosixPath
                if not self._builder_config.process_video:
                    datum["exampleVideo"] = str(datum["exampleVideo"])

        if self._builder_config.include_pose is not None:
            poses_dir = dl_manager.download_and_extract(_POSE_URLS[self._builder_config.include_pose])
            poses_dir = poses_dir.joinpath("signsuisse")
            # Some poses are corrupted. We need to remove them for now.
            # TODO: rerun these poses
            bad_poses = [
                'ssdca608e11c025737cb31eba18c88ab50.pose',
                'ss703cc76745bf64ad09da82be3052be0c.pose',
                'ss00f14fcf3240806f1f375e24751bbcda.pose'
            ]
            for bad_pose in bad_poses:
                if poses_dir.joinpath(bad_pose).exists():
                    poses_dir.joinpath(bad_pose).unlink()

            id_func = lambda opt: 'ss' + hashlib.md5(("signsuisse" + opt[0] + opt[1]).encode()).hexdigest()

            for datum in data:
                pose_file = poses_dir.joinpath(id_func([datum["id"], "isolated"]) + ".pose")
                datum["pose"] = pose_file if pose_file.exists() else None

                if datum["exampleVideo"] != "":
                    pose_file = poses_dir.joinpath(id_func([datum["id"], "example"]) + ".pose")
                    datum["examplePose"] = pose_file if pose_file.exists() else None
                else:
                    datum["examplePose"] = None

        return {"train": self._generate_examples(data)}

    def _generate_examples(self, data):
        for datum in data:
            yield datum["id"], datum
