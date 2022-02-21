"""SignBank dataset."""
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from ...datasets import SignDatasetConfig

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

PUDDLES = {
    2: ["my", "mm"],  # Myanmar Dictionary",
    4: ["en", "us"],  # Dictionary US",
    5: ["en", "us"],  # Literature US",
    11: ["en", "sg"],  # Singapore Sign Language (SgSL) Dictionary",
    12: ["zh-CN", "hk"],  # Hong Kong Dictionary",
    13: ["zh-CN", "hk"],  # Hong Kong Literature",
    14: ["en", "sg"],  # Singapore Sign Language (SgSL) Literature",
    16: ["es", "hn"],  # Diccionario Honduras",
    17: ["", ""],  # Deaf Harbor",
    18: ["am", "et"],  # Dictionary Ethiopia",
    19: ["pl", "pl"],  # Słownik PL",
    20: ["fr", "ch"],  # Littérature CH-fr",
    21: ["en", "us"],  # Encyclopedia US",
    22: ["fr", "ch"],  # Encyclopédie CH-fr",
    23: ["no", "no"],  # Litteratur NO",
    24: ["no", "no"],  # Leksikon NO",
    25: ["en", "us"],  # LLCN & SignTyp",
    26: ["de", "de"],  # Literatur DE",
    27: ["de", "de"],  # Enzyklopädie DE",
    28: ["en", "us"],  # ASL Bible Dictionary",
    29: ["de", "at"],  # Wörterbuch AT",
    30: ["da", "dk"],  # Ordbog Danmark",
    31: ["mt", "mt"],  # Dictionary Malta",
    32: ["en", "ng"],  # Dictionary Nigeria",
    33: ["pt", "pt"],  # Dicionário Portugal",
    34: ["th", "th"],  # Dictionary Thailand",
    35: ["en", "isl"],  # Dictionary International",
    36: ["cs", "cz"],  # Literatura CZ",
    37: ["cs", "cz"],  # Encyklopedie CZ",
    38: ["pl", "pl"],  # Literatura PL",
    39: ["pl", "pl"],  # Encyklopedie PL",
    40: ["ar", "sa"],  # Dictionary Saudi Arabia",
    41: ["es", "ar"],  # Diccionario Argentina",
    42: ["en", "au"],  # Dictionary Australia",
    43: ["fr", "be"],  # Dictionnaire BE-fr",
    44: ["nl", "be"],  # Woordenboek Flanders",
    45: ["es", "bo"],  # Diccionario Bolivia",
    46: ["pt", "br"],  # Dicionário Brasil",
    47: ["fr", "ca"],  # Dictionnaire Quebec",
    48: ["de", "ch"],  # Wörterbuch CH-de",
    49: ["fr", "ch"],  # Dictionnaire CH-fr",
    50: ["it", "ch"],  # Dizionario CH-it",
    51: ["es", "co"],  # Diccionario Colombia",
    52: ["sk", "sk"],  # Slovník CZ",
    53: ["de", "de"],  # Wörterbuch DE",
    54: ["", ""],  # Vortaro",
    55: ["es", "es"],  # Diccionario España",
    56: ["ca", "es"],  # Diccionario Catalán",
    57: ["fi", "fi"],  # Dictionary Finland",
    58: ["fr", "fr"],  # Dictionnaire FR",
    59: ["en", "gb"],  # Dictionary Great Britain",
    60: ["en", "ie"],  # Dictionary Northern Ireland",
    61: ["gr", "gr"],  # Dictionary Greece",
    62: ["en", "ir"],  # Dictionary Ireland",
    63: ["it", "it"],  # Dizionario IT",
    64: ["ja", "jp"],  # Dictionary Japan",
    65: ["es", "mx"],  # Diccionario Mexico",
    66: ["ms", "my"],  # Dictionary Malaysia",
    67: ["es", "ni"],  # Diccionario Nicaragua",
    68: ["nl", "nl"],  # Woordenboek NL",
    69: ["", ""],  # Ordbok NO",
    70: ["en", "nz"],  # Dictionary New Zealand",
    71: ["es", "pe"],  # Diccionario Peru",
    72: ["fil", "ph"],  # Dictionary Philippines",
    73: ["sv", "se"],  # Ordbok Sverige",
    74: ["sl", "sl"],  # Slovar Slovenia",
    75: ["zh-tw", "tw"],  # Dictionary Taiwan",
    76: ["es", "ve"],  # Diccionario Venezuela",
    77: ["en", "za"],  # Dictionary South Africa",
    78: ["ko", "kr"],  # Dictionary Korea",
    79: ["sw", "ke"],  # Dictionary Kenya",
    80: ["", ""],  # Project 2 Dictionary Sorting",
    81: ["fr", "ca"],  # Littérature Quebec",
    82: ["sq", "al"],  # Dictionary Albania",
    83: ["zh-cn", "cn"],  # Dictionary China",
    84: ["ar", "eg"],  # Dictionary Egypt",
    85: ["hi", "in"],  # Dictionary India",
    86: ["ar", "jo"],  # Dictionary Jordan",
    87: ["ur", "pk"],  # Dictionary Pakistan",
    88: ["ru", "ru"],  # Dictionary Russia",
    89: ["sk", "sk"],  # Dictionary Slovakia",
    90: ["tr", "tr"],  # Dictionary Turkey",
    91: ["ar", "sa"],  # Literature Saudi Arabia",
    92: ["ar", "jo"],  # Literature Jordan",
    93: ["es", "es"],  # Literatura España",
    94: ["ca", "es"],  # Literatura Catalán",
    95: ["fr", "be"],  # Littérature BE-fr",
    96: ["de", "ch"],  # Literatur CH-de",
    98: ["nl", "be"],  # Literatuur Flanders",
    99: ["ja", "jp"],  # Literature Japan",
    100: ["am", "et"],  # Literature Ethiopia",
    103: ["mt", "mt"],  # Malta LSM Private Puddle",
    104: ["ar", "tn"],  # Dictionnaire Tunisien",
    105: ["", ""],  # DAC Private Puddle",
    106: ["ps", "af"],  # Dictionary Afghanistan",
    107: ["lt", "lt"],  # Dictionary Lithuania",
    108: ["lv", "lv"],  # Dictionary Latvia",
    109: ["et", "et"],  # Dictionary Estonia",
    110: ["he", "il"],  # Dictionary Israel",
    111: ["", ""],  # Project 1 Translate Wiki",
    112: ["es", "gt"],  # Dictionary Guatemala",
    113: ["ht", "ht"],  # Dictionary Haiti",
    114: ["pt", "br"],  # Literatura Brasil",
    115: ["pt", "pt"],  # Literatura Portugal",
    116: ["pt", "br"],  # Enciclopédia Brasil",
    117: ["pt", "pt"],  # Enciclopédia Portugal",
    118: ["da", "dk"],  # Litteratur Danmark",
    119: ["es", "ni"],  # Literatura Nicaragua",
    120: ["es", "mx"],  # Literatura Mexico",
    122: ["hu", "hu"],  # Dictionary Hungary",
    123: ["hu", "hu"],  # Literature Hungary",
    124: ["fr", "fr"],  # Literature France",
    125: ["en", "gb"],  # Literature Great Britain",
    126: ["ar", "tn"],  # Littérature Tunisien",
    127: ["mt", "mt"],  # Literature Malta",
    128: ["mw", "mw"],  # Dictionary Malawi",
    129: ["gn", "py"],  # Diccionario Paraguay",
    130: ["uk", "ua"],  # Dictionary Ukraine",
    131: ["", ""],  # Ordabók IS",
    132: ["ro", "ro"],  # Dictionary Romania",
    133: ["ne", "np"],  # Dictionary Nepal",
    134: ["bg", "bg"],  # Dictionary Bulgaria",
    135: ["es", "cl"],  # Diccionario Chile",
    136: ["es", "ec"],  # Diccionario Ecuador",
    137: ["es", "sv"],  # Diccionario El Salvador",
    138: ["ro", "ro"],  # Literature Romania",
    139: ["ro", "ro"],  # Encyclopedia Romania",
    140: ["fr", "ca"],  # Encyclopédie Quebec",
    141: ["ru", "ru"],  # Literature Russia",
    142: ["ru", "ru"],  # Encyclopedia Russia",
    143: ["es", "uy"],  # Diccionario Uruguay",
    144: ["es", "uy"],  # Literatura Uruguay",
    145: ["es", "ar"],  # Literatura Argentina",
    146: ["es", "ar"],  # Enciclopedia Argentina",
    147: ["mt", "mt"],  # Literature Malta Archive",
    148: ["sl", "sl"],  # Besedila Slovenia",
    149: ["sl", "sl"],  # Enciklopedija Slovenia",
    150: ["", ""],  # Anthropology Book Project",
    151: ["en", "us"],  # ASL Bible Books NLT",
    152: ["en", "us"],  # ASL Bible Books Shores Deaf Church",
    153: ["vn", "vn"],  # Dictionary Vietnam"
}


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
            "assumed_spoken_language_code": tfds.features.Text(),
            "country_code": tfds.features.Text(),
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
                        assumed_spoken_language_code, country_code = PUDDLES[puddle] if puddle in PUDDLES else ["", ""]
                        i += 1
                        yield sample_id, {
                            'puddle': puddle,
                            'id': _id,
                            'assumed_spoken_language_code': assumed_spoken_language_code,
                            'country_code': country_code,
                            "created_date": str(datetime.fromtimestamp(cdt)),
                            "modified_date": str(datetime.fromtimestamp(mdt)),
                            "sign_writing": texts[0],
                            "terms": texts[1:],
                            "user": usr
                        }
