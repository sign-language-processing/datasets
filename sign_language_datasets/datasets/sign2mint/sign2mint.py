"""A German Sign Language (DGS) lexicon for natural science subjects."""
import json

import tensorflow as tf
import tensorflow_datasets as tfds

from sign_language_datasets.datasets import SignDatasetConfig

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
                "symbolIds": tfds.features.Sequence(tfds.features.Text())
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
            with open(annotations_path, "r", encoding="utf-8")  as f:
                annotations = json.load(f)
                videos = [a["videoLink"] for a in annotations]
                video_paths = dl_manager.download(videos)
                local_videos = {k: v for k, v in zip(videos, video_paths)}

        return {
            'train': self._generate_examples(annotations_path, local_videos)
        }

    def _generate_examples(self, annotations_path, local_videos):
        """Yields examples."""

        with open(annotations_path, "r", encoding="utf-8")  as f:
            data = json.load(f)
            for datum in data:
                del datum["empfehlung"]  # remove unused property

                video_link = datum["videoLink"]
                del datum["videoLink"]

                datum["video"] = local_videos[video_link] if video_link in local_videos else video_link
                datum["gebaerdenschrift"]["symbolIds"] = [s["symbolId"] for s in datum["gebaerdenschrift"]["symbolIds"]]

                yield datum["id"], datum
