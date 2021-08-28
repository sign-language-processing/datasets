"""SignBank dataset."""
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from sign_language_datasets.datasets import SignDatasetConfig

_DESCRIPTION = """
SignBank Site: SignWriting Software for Sign Languages, including SignMaker 2017, 
SignPuddle Online, the SignWriting Character Viewer, SignWriting True Type Fonts, 
Delegs SignWriting Editor, SignBank Databases in FileMaker, SignWriting DocumentMaker, 
SignWriting Icon Server, the International SignWriting Alphabet (ISWA 2010) HTML Reference Guide, 
the ISWA 2010 Font Reference Library and the RAND Keyboard for SignWriting.
"""

# TODO(SignBank): BibTeX citation
_CITATION = """
"""


class SignBank(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for SignBank dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=False)
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {
            "puddle": tf.int32,
            "id": tfds.features.Text(),
            "created_date": tfds.features.Text(),
            "modified_date": tfds.features.Text(),
            "sign_writing": tfds.features.Text(),
            "terms": tfds.features.Sequence(tfds.features.Text()),
            "user": tfds.features.Text(),
        }

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="http://signbank.org/",
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        index = dl_manager.download("http://signbank.org/signpuddle2.0/data/spml/")
        regex = r"\"sgn[\d]+.spml\""
        with open(index, "r", encoding="utf-8") as f:
            matches = [match.group()[1:-1] for match in re.finditer(regex, f.read(), re.MULTILINE)]
            spml_urls = ["http://signbank.org/signpuddle2.0/data/spml/" + match
                         for match in matches if match != "sgn0.spml"]

        spmls = dl_manager.download(spml_urls)

        return {
            'train': self._generate_examples(spmls),
        }

    def _generate_examples(self, spmls: List[str]):
        """Yields examples."""

        i = 0
        for spml in tqdm(spmls):
            tree = ET.parse(spml)
            root = tree.getroot()

            puddle = int(root.attrib['puddle'])

            children = root.getchildren()
            for child in children:
                if child.tag == "entry":
                    _id = child.attrib['id']
                    mdt = int(child.attrib['mdt']) if 'mdt' in child.attrib and child.attrib['mdt'] != '' else 0
                    cdt = int(child.attrib['cdt']) if 'cdt' in child.attrib and child.attrib['cdt'] != '' else mdt
                    usr = child.attrib['usr'] if 'usr' in child.attrib else ''
                    texts = [c.text for c in child.getchildren() if c.tag != "src" and c.text is not None]

                    # print(child.attrib)
                    # print(texts)
                    if len(texts) > 0:
                        sample_id = "_".join([str(puddle), _id, str(i)])
                        i += 1
                        yield sample_id, {
                            'puddle': puddle,
                            'id': _id,
                            "created_date": str(datetime.fromtimestamp(cdt)),
                            "modified_date": str(datetime.fromtimestamp(mdt)),
                            "sign_writing": texts[0],
                            "terms": texts[1:],
                            "user": usr
                        }
