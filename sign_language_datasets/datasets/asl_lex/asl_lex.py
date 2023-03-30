"""A database of lexical and phonological properties of American Sign Language signs. """
from os import path
import tensorflow_datasets as tfds
import csv

from ..warning import dataset_warning
from ...datasets.config import SignDatasetConfig

_DESCRIPTION = """
ASL-LEX is a database of lexical and phonological properties of American Sign Language signs. 
It was first released in 2016 with nearly 1,000 signs. 
ASL-LEX was updated in Fall 2020 with greatly expanded information and an increased size of 2,723 signs.
"""

_CITATION = """
@article{caselli2017asl,
  title={ASL-LEX: A lexical database of American Sign Language},
  author={Caselli, Naomi K and Sehyr, Zed Sevcikova and Cohen-Goldberg, Ariel M and Emmorey, Karen},
  journal={Behavior research methods},
  volume={49},
  pages={784--801},
  year={2017},
  publisher={Springer}
}
"""

header_file = path.join(path.dirname(path.realpath(__file__)), "data-key.csv")


class AslLex(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for asl-lex dataset."""

    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release with nearly 1,000 signs.",
        "2.0.0": "greatly expanded information and an increased size of 2,723 signs",
    }

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=True),
        SignDatasetConfig(name="annotations", include_video=False),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        # Open the CSV file
        with open(header_file, "r", encoding="utf-8") as csvfile:
            # Create a CSV reader object
            reader = csv.DictReader(csvfile)
            self.features = {row["VariableName(CSV&OSF)"]: tfds.features.Text() for row in reader}

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(self.features),
            homepage="https://asl-lex.org/",
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        dataset_warning(self)

        csv_path = dl_manager.download_and_extract("https://osf.io/download/9nygd/")

        return {"train": self._generate_examples(csv_path)}

    def _generate_examples(self, csv_path: str):
        """Yields examples."""

        with open(csv_path, "r", encoding="ISO-8859-1") as csvfile:
            # Create a CSV reader object
            reader = csv.DictReader(csvfile)

            for i, row in enumerate(reader):
                new_row = {f: row[f] for f in self.features}
                yield str(i), new_row
