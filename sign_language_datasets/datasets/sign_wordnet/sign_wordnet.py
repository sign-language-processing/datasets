"""The Multilingual Sign Language Wordnet"""
import csv

import tensorflow_datasets as tfds

from ..warning import dataset_warning
from ...datasets import SignDatasetConfig

_DESCRIPTION = """
This is an ongoing project to add several European sign languages to the Open Multilingual Wordnet. It is part of the ongoing EASIER project, whose goal is to create an intelligent translation framework for sign languages. This Wordnet aims at connecting signs with a shared meaning accross languages.

The current available version covers German Sign Language, Greek Sign Language, British Sign Language, Sign Language of the Netherlands, and French Sign Language. More than 9 000 signs have been successfully linked to one or more synsets, with 15 000 more still being processed. It is a work in progress, and the coverage is limited.
"""

_CITATION = """
@inproceedings{Bigeard2022IntroducingSL,
  title={Introducing Sign Languages to a Multilingual Wordnet: Bootstrapping Corpora and Lexical Resources of Greek Sign Language and German Sign Language},
  author={Sam Bigeard and Marc Schulder and Maria Kopf and Thomas Hanke and Kyriaki Vasilaki and Anna Vacalopoulou and Theodoros Goulas and Lida Dimou and Evita F. Fotinea and Eleni Efthimiou},
  booktitle={SIGNLANG},
  year={2022},
  url={https://api.semanticscholar.org/CorpusID:250086794}
}
"""

# Map between Sign Wornet language codes and IANA language codes
IANA_MAP = {
    "bsl": "bfi",
    "dgs": "gsg",
    "gsl": "gss",
    "lsf": "fsl",
    "ngt": "dse"
}


def no_space(d: dict):
    for k, v in d.items():
        if isinstance(v, str):
            d[k] = v.strip()
    return d

class SignWordnet(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for SignWordnet dataset."""

    VERSION = tfds.core.Version("0.2.0")
    RELEASE_NOTES = {
        "0.2.0": "Initial release.",
    }

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=True)
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {
            "id": tfds.features.Text(),
            "sign_language": tfds.features.Text(),
            "sign_language_name": tfds.features.Text(),
            "translations": tfds.features.Sequence({
                "language": tfds.features.Text(),
                "text": tfds.features.Text()
            }),
            "links": tfds.features.Text(),
            "gloss": tfds.features.Text(),
            "video": tfds.features.Text(),
            "synset_id": tfds.features.Text(),
            "confidence": tfds.features.Text(),
        }

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="https://www.sign-lang.uni-hamburg.de/easier/sign-wordnet/index.html",
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dataset_warning(self)

        try:
            import nltk
        except ImportError:
            raise ImportError("Please install nltk with: pip install nltk")

        nltk.download("wordnet")
        nltk.download("omw-1.4")
        nltk.download("extended_omw")

        signs_data_url = "https://www.sign-lang.uni-hamburg.de/easier/sign-wordnet/static/csv/sign_data.csv"
        accepted_synsets_url = "https://www.sign-lang.uni-hamburg.de/easier/sign-wordnet/static/csv/synset_links_accepted.csv"
        custom_synsets_url = "https://www.sign-lang.uni-hamburg.de/easier/sign-wordnet/static/csv/custom_synsets.csv"

        signs_data, accepted_synsets, custom_synsets = dl_manager.download(
            [signs_data_url, accepted_synsets_url, custom_synsets_url])

        return {
            "train": self._generate_examples(signs_data, accepted_synsets, custom_synsets),
        }

    def _generate_examples(self, signs_data: str, accepted_synsets: str, custom_synsets: str):
        from nltk.corpus import wordnet as wn
        # List all available languages in OMW
        languages = wn.langs()
        print('Translation Languages', languages)

        with open(custom_synsets, "r", encoding="utf-8") as csvfile:
            custom_synsets_dict = {row["synset_id"]: no_space(row) for row in csv.DictReader(csvfile)}

        with open(signs_data, "r", encoding="utf-8") as csvfile:
            signs_data_dict = {row["sign_id_wordnet"]: no_space(row) for row in csv.DictReader(csvfile)}

        with open(accepted_synsets, "r", encoding="utf-8") as csvfile:
            accepted_synsets_dict = {row["sign_id"]: no_space(row) for row in csv.DictReader(csvfile)}

        for _id, row in signs_data_dict.items():
            accepted_synset = accepted_synsets_dict.get(_id, {})
            synset_id = accepted_synset.get("synset_id", "")  # Looks like  omw.00672433-v
            translations = []

            if synset_id != "":  # Looks like  omw.00672433-v
                if synset_id in custom_synsets_dict:
                    translations.append({
                        "language": "en",
                        "text": custom_synsets_dict[synset_id]["words_english"]
                    })
                else:
                    # TODO: track open issue https://github.com/omwn/omw-data/issues/35
                    _, omw_id = synset_id.split('.')
                    omw_offset, omw_pos = omw_id.split('-')
                    synset = wn.synset_from_pos_and_offset(omw_pos, int(omw_offset))

                    # Get translations in all languages
                    for lang in languages:
                        lemmas = synset.lemmas(lang=lang)
                        if lemmas:  # If there are lemmas for this language
                            for lemma in lemmas:
                                translations.append({"language": lang, "text": lemma.name()})

            yield _id, {
                "id": _id,
                "sign_language": IANA_MAP[row["language"]],
                "sign_language_name": row["language"],
                "translations": translations,
                "links": row["links"],
                "gloss": row["gloss"],
                "video": row["video"],
                "synset_id": accepted_synset.get("synset_id", ""),
                "confidence": accepted_synset.get("confidence", ""),
            }
