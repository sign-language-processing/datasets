"""A German Sign Language (DGS) lexicon for natural science subjects."""
import json
import os

import tensorflow as tf
import tensorflow_datasets as tfds

from ...datasets import SignDatasetConfig
import urllib.request
from cv2 import cv2

from ...utils.signwriting.ocr import image_to_fsw

_DESCRIPTION = """
The specialist signs developed in the project break down barriers for deaf people 
and thus facilitate their access to education in natural science subjects. 
Improved communication enables deaf people in the MINT subjects to participate better in research and science, 
from school to university to independent work. With the MINT specialist sign language dictionary, 
you have a better chance of developing your scientific ideas as a researcher; 
it makes work in the laboratory easier and enriches your own career planning, as research results, 
for example, can be better communicated. For the first time, teachers, 
students and pupils have access to a uniform and constantly expanding MINT specialist sign language dictionary,
"""

# TODO(sign2mint): BibTeX citation
_CITATION = """
"""

OCR_CACHE_PATH = os.path.join(os.path.dirname(__file__), "ocr_cache.txt")


class Sign2MINT(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for sign2mint dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=True),
        SignDatasetConfig(name="annotations", include_video=False),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {
            "id": tfds.features.Text(),
            "fachbegriff": tfds.features.Text(),
            "fachgebiete": tfds.features.Sequence(tfds.features.Text()),
            "ursprung": tfds.features.Sequence(tfds.features.Text()),
            "verwendungskontext": tfds.features.Sequence(tfds.features.Text()),
            "definition": tfds.features.Text(),
            "bedeutungsnummern": tfds.features.Text(),
            "wortlink": tfds.features.Text(),
            "wikipedialink": tfds.features.Text(),
            "otherlink": tfds.features.Text(),
            "variants": tf.int32,
            "gebaerdenschrift": {
                "url": tfds.features.Text(),
                "symbolIds": tfds.features.Sequence(tfds.features.Text()),
                "fsw": tfds.features.Text()
            }
        }

        if self._builder_config.include_video and self._builder_config.process_video:
            features["video"] = self._builder_config.video_feature((1920, 1080))
        else:
            features["video"] = tfds.features.Text()

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="https://sign2mint.de/",
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        annotations_path = dl_manager.download("https://sign2mint.de/api/entries/all/")

        local_videos = {}
        if self._builder_config.include_video and self._builder_config.process_video:
            with open(annotations_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
                videos = [a["videoLink"] for a in annotations]
                video_paths = dl_manager.download(videos)
                local_videos = {k: v for k, v in zip(videos, video_paths)}

        return {
            'train': self._generate_examples(annotations_path, local_videos)
        }

    ocr_cache = None

    def _ocr(self, datum):
        if self.ocr_cache is None:
            with open(OCR_CACHE_PATH, "r") as f:
                lines = [l.split(" ") for l in f.readlines()]
                self.ocr_cache = {l[0]: l[1] for l in lines}

        image_url = datum['gebaerdenschrift']['url'].replace("&transparent=true", "")
        if image_url in self.ocr_cache:
            return self.ocr_cache[image_url]

        urllib.request.urlretrieve(image_url, "sign.png")
        img_rgb = cv2.imread('sign.png')

        symbols = datum['gebaerdenschrift']['symbolIds']

        fsw = image_to_fsw(img_rgb, symbols)

        with open(OCR_CACHE_PATH, "a") as f:
            f.write(image_url + " " + fsw + "\n")

        return fsw

    def _generate_examples(self, annotations_path, local_videos):
        """Yields examples."""

        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for datum in data:
                del datum["empfehlung"]  # remove unused property

                video_link = datum["videoLink"]
                del datum["videoLink"]

                datum["video"] = local_videos[video_link] if video_link in local_videos else video_link
                datum["gebaerdenschrift"]["symbolIds"] = [s["symbolKey"] for s in datum["gebaerdenschrift"]["symbolIds"]
                                                          if s["symbolKey"] != ""]
                datum["gebaerdenschrift"]["fsw"] = self._ocr(datum)

                yield datum["id"], datum
