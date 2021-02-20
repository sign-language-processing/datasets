"""ASLG-PC12: Synthetic English-ASL Gloss Parallel Corpus 2012"""

import tensorflow_datasets as tfds
from tensorflow.io.gfile import GFile

_DESCRIPTION = """
A large synthetic collection of parallel English and ASL-Gloss texts.
There are two string features: text, and gloss.
"""

_CITATION = """
@inproceedings{othman2012english,
  title={English-asl gloss parallel corpus 2012: Aslg-pc12},
  author={Othman, Achraf and Jemni, Mohamed},
  booktitle={5th Workshop on the Representation and Processing of Sign Languages: Interactions between Corpus and Lexicon LREC},
  year={2012}
}
"""

_GLOSS_URL = "https://www.achrafothman.net/aslsmt/corpus/sample-corpus-asl-en.asl"
_TEXT_URL = "https://www.achrafothman.net/aslsmt/corpus/sample-corpus-asl-en.en"


class AslgPc12(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for aslg_pc12 dataset."""

    VERSION = tfds.core.Version("0.0.1")
    RELEASE_NOTES = {"0.0.1": "Sample of the full corpus"}

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({"gloss": tfds.features.Text(), "text": tfds.features.Text()}),
            supervised_keys=("gloss", "text"),  # Set to `None` to disable
            homepage="https://achrafothman.net/site/asl-smt/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        gloss_path, text_path = dl_manager.download([_GLOSS_URL, _TEXT_URL])

        return {
            "train": self._generate_examples(gloss_path, text_path),
        }

    def _generate_examples(self, gloss_path: str, text_path: str):
        """Yields examples."""

        with GFile(gloss_path, "r") as gloss_f:
            with GFile(text_path, "r") as text_f:
                for i, (gloss, text) in enumerate(zip(gloss_f, text_f)):
                    yield i, {"gloss": gloss, "text": text}
