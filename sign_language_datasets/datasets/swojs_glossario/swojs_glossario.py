"""A Brazilian Sign Language (Libras) lexicon."""
import json

import tensorflow as tf
import tensorflow_datasets as tfds

from ...datasets import SignDatasetConfig

_DESCRIPTION = """
A lexicon for Brazilian Sign Language (Libras), including videos, and SignWriting transcription.
"""

# TODO(swojs_glossario): BibTeX citation
_CITATION = """
"""


def get_media_id(datum):
    if "o:media" not in datum or len(datum["o:media"]) == 0:
        return None
    return datum["o:media"][0]["o:id"]


def get_transcription(datum):
    vals = []
    if "oc:transcription" in datum:
        vals = datum["oc:transcription"]
    elif "http://purl.org/linguistics/gold:writtenRealization" in datum:
        vals = datum["http://purl.org/linguistics/gold:writtenRealization"]

    return [t["@value"] for t in vals]


class SwojsGlossario(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for swojs_glossario dataset."""

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
            "id": tf.int32,
            "title": tfds.features.Text(),
            "created_date": tfds.features.Text(),
            "modified_date": tfds.features.Text(),
            "sign_writing": tfds.features.Sequence(tfds.features.Text()),
            "spoken_language": tfds.features.Sequence(tfds.features.Text()),
            "signed_language": tfds.features.Sequence(tfds.features.Text())
        }

        if self._builder_config.include_video and self._builder_config.process_video:
            features["video"] = self._builder_config.video_feature((1920, 1080))
        else:
            features["video"] = tfds.features.Text()

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="http://swojs.ibict.br/portal/s/swojs/page/glossario",
            supervised_keys=None,
            citation=_CITATION,
        )

    def get_media_dict(self, media_path):
        with open(media_path, "r", encoding="utf-8")  as g:
            media = json.load(g)
            return {m["o:id"]: m["o:original_url"] for m in media}

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        annotations_path, media_path = dl_manager.download([
            "http://swojs.ibict.br/portal/api/items?page",
            "http://swojs.ibict.br/portal/api/media?page"
        ])

        media_dict = self.get_media_dict(media_path)

        local_videos = {}
        if self._builder_config.include_video and self._builder_config.process_video:
            with open(annotations_path, "r", encoding="utf-8")  as f:
                annotations = json.load(f)

                medias = [_id for _id in [get_media_id(a) for a in annotations] if _id is not None]
                media_urls = [media_dict[_id] for _id in medias]

                video_paths = dl_manager.download(media_urls)
                local_videos = {k: v for k, v in zip(medias, video_paths)}

        return {
            'train': self._generate_examples(annotations_path, local_videos, media_dict)
        }

    def _generate_examples(self, annotations_path, local_videos, media_dict):
        """Yields examples."""

        with open(annotations_path, "r", encoding="utf-8")  as f:
            data = json.load(f)
            for datum in data:
                features = {
                    "id": datum["o:id"],
                    "title": datum["o:title"] if datum["o:title"] is not None else "",
                    "created_date": datum["o:created"]["@value"],
                    "modified_date": datum["o:modified"]["@value"],
                    "sign_writing": get_transcription(datum),
                    "spoken_language": [],
                    "signed_language": []
                }

                if "http://www.lingvoj.org/ontology:originalLanguage" in datum:
                    features["spoken_language"] = [t["@value"] for t in
                                                   datum["http://www.lingvoj.org/ontology:originalLanguage"]]

                if "http://www.lingvoj.org/ontology:targetLanguage" in datum:
                    features["signed_language"] = [t["@value"] for t in
                                                   datum["http://www.lingvoj.org/ontology:targetLanguage"]]

                media_id = get_media_id(datum)
                if media_id in local_videos:
                    features["video"] = local_videos[media_id]
                elif media_id is not None:
                    features["video"] = media_dict[media_id]
                else:
                    features["video"] = ""

                yield str(features["id"]), features
