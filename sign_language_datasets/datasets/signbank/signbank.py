"""SignBank dataset."""
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from ..warning import dataset_warning
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


def is_signwriting(fsw: str) -> bool:
    return bool(re.match(r"[A]?[abcdef0-9S]*?([BLMR])(\d{3})x(\d{3})", fsw))


PUDDLES = {
    2: ["my", "mm", "ysm"],  # Myanmar Dictionary
    4: ["en", "us", "ase"],  # Dictionary US
    5: ["en", "us", "ase"],  # Literature US
    11: ["en", "sg", "sls"],  # Singapore Sign Language (SgSL) Dictionary
    12: ["zh-CN", "hk", "hks"],  # Hong Kong Dictionary
    13: ["zh-CN", "hk", "hks"],  # Hong Kong Literature
    14: ["en", "sg", "sls"],  # Singapore Sign Language (SgSL) Literature
    16: ["es", "hn", "hds"],  # Diccionario Honduras
    17: ["en", "us", "ase"],  # Deaf Harbor
    18: ["am", "et", "eth"],  # Dictionary Ethiopia
    19: ["pl", "pl", "pso"],  # Słownik PL
    20: ["fr", "ch", "ssr"],  # Littérature CH-fr
    21: ["en", "us", "ase"],  # Encyclopedia US
    22: ["fr", "ch", "ssr"],  # Encyclopédie CH-fr
    23: ["no", "no", "nsl"],  # Litteratur NO
    24: ["no", "no", "nsl"],  # Leksikon NO
    25: ["en", "us", "ase"],  # LLCN & SignTyp
    26: ["de", "de", "gsg"],  # Literatur DE
    27: ["de", "de", "gsg"],  # Enzyklopädie DE
    28: ["en", "us", "ase"],  # ASL Bible Dictionary
    29: ["de", "at", "asq"],  # Wörterbuch AT
    30: ["da", "dk", "dsl"],  # Ordbog Danmark
    31: ["mt", "mt", "mdl"],  # Dictionary Malta
    32: ["en", "ng", "nsi"],  # Dictionary Nigeria
    33: ["pt", "pt", "psr"],  # Dicionário Portugal
    34: ["th", "th", "tsq"],  # Dictionary Thailand
    35: ["en", "isl", "ase"],  # Dictionary International
    36: ["cs", "cz", "cse"],  # Literatura CZ
    37: ["cs", "cz", "cse"],  # Encyklopedie CZ
    38: ["pl", "pl", "pso"],  # Literatura PL
    39: ["pl", "pl", "pso"],  # Encyklopedie PL
    40: ["ar", "sa", "sdl"],  # Dictionary Saudi Arabia
    41: ["es", "ar", "aed"],  # Diccionario Argentina
    42: ["en", "au", "asf"],  # Dictionary Australia
    43: ["fr", "be", "sfb"],  # Dictionnaire BE-fr
    44: ["nl", "be", "vgt"],  # Woordenboek Flanders
    45: ["es", "bo", "bvl"],  # Diccionario Bolivia
    46: ["pt", "br", "bzs"],  # Dicionário Brasil
    47: ["fr", "ca", "fcs"],  # Dictionnaire Quebec
    48: ["de", "ch", "sgg"],  # Wörterbuch CH-de
    49: ["fr", "ch", "ssr"],  # Dictionnaire CH-fr
    50: ["it", "ch", "slf"],  # Dizionario CH-it
    51: ["es", "co", "csn"],  # Diccionario Colombia
    52: ["sk", "sk", "svk"],  # Slovník CZ
    53: ["de", "de", "gsg"],  # Wörterbuch DE
    54: ["eo", "", "ase"],  # Vortaro (esperanto)
    55: ["es", "es", "ssp"],  # Diccionario España
    56: ["ca", "es", "csc"],  # Diccionario Catalán
    57: ["fi", "fi", "fse"],  # Dictionary Finland
    58: ["fr", "fr", "fsl"],  # Dictionnaire FR
    59: ["en", "gb", "bfi"],  # Dictionary Great Britain
    60: ["en", "ie", "isg"],  # Dictionary Northern Ireland
    61: ["gr", "gr", "gss"],  # Dictionary Greece
    62: ["en", "ir", "psc"],  # Dictionary Ireland
    63: ["it", "it", "ise"],  # Dizionario IT
    64: ["ja", "jp", "jsl"],  # Dictionary Japan
    65: ["es", "mx", "mfs"],  # Diccionario Mexico
    66: ["ms", "my", "xml"],  # Dictionary Malaysia
    67: ["es", "ni", "ncs"],  # Diccionario Nicaragua
    68: ["nl", "nl", "dse"],  # Woordenboek NL
    69: ["no", "no", "nsl"],  # Ordbok NO
    70: ["en", "nz", "nzs"],  # Dictionary New Zealand
    71: ["es", "pe", "prl"],  # Diccionario Peru
    72: ["fil", "ph", "psp"],  # Dictionary Philippines
    73: ["sv", "se", "swl"],  # Ordbok Sverige
    74: ["sl", "sl", "ysl"],  # Slovar Slovenia
    75: ["zh-TW", "tw", "tss"],  # Dictionary Taiwan
    76: ["es", "ve", "vsl"],  # Diccionario Venezuela
    77: ["en", "za", "sfs"],  # Dictionary South Africa
    78: ["ko", "kr", "kvk"],  # Dictionary Korea
    79: ["sw", "ke", "xki"],  # Dictionary Kenya
    80: ["pt", "pt", "psr"],  # Project 2 Dictionary Sorting
    81: ["fr", "ca", "fcs"],  # Littérature Quebec
    82: ["sq", "al", "sqk"],  # Dictionary Albania
    83: ["zh-CN", "cn", "csl"],  # Dictionary China
    84: ["ar", "eg", "esl"],  # Dictionary Egypt
    85: ["hi", "in", "ins"],  # Dictionary India
    86: ["ar", "jo", "jos"],  # Dictionary Jordan
    87: ["ur", "pk", "pks"],  # Dictionary Pakistan
    88: ["ru", "ru", "rsl"],  # Dictionary Russia
    89: ["sk", "sk", "svk"],  # Dictionary Slovakia
    90: ["tr", "tr", "tsm"],  # Dictionary Turkey
    91: ["ar", "sa", "sdl"],  # Literature Saudi Arabia
    92: ["ar", "jo", "jos"],  # Literature Jordan
    93: ["es", "es", "ssp"],  # Literatura España
    94: ["ca", "es", "csc"],  # Literatura Catalán
    95: ["fr", "be", "sfb"],  # Littérature BE-fr
    96: ["de", "ch", "sgg"],  # Literatur CH-de
    98: ["nl", "be", "vgt"],  # Literatuur Flanders
    99: ["ja", "jp", "jsl"],  # Literature Japan
    100: ["am", "et", "eth"],  # Literature Ethiopia
    103: ["mt", "mt", "mdl"],  # Malta LSM Private Puddle
    104: ["ar", "tn", "tse"],  # Dictionnaire Tunisien
    105: ["en", "us", "ase"],  # DAC Private Puddle
    106: ["ps", "af", "afg"],  # Dictionary Afghanistan
    107: ["lt", "lt", "lls"],  # Dictionary Lithuania
    108: ["lv", "lv", "lsl"],  # Dictionary Latvia
    109: ["et", "et", "eso"],  # Dictionary Estonia
    110: ["he", "il", "isr"],  # Dictionary Israel
    111: ["en", "us", "ase"],  # Project 1 Translate Wiki
    112: ["es", "gt", "gsm"],  # Dictionary Guatemala
    113: ["ht", "ht", ""],  # Dictionary Haiti
    114: ["pt", "br", "bzs"],  # Literatura Brasil
    115: ["pt", "pt", "psr"],  # Literatura Portugal
    116: ["pt", "br", "bzs"],  # Enciclopédia Brasil
    117: ["pt", "pt", "psr"],  # Enciclopédia Portugal
    118: ["da", "dk", "dsl"],  # Litteratur Danmark
    119: ["es", "ni", "ncs"],  # Literatura Nicaragua
    120: ["es", "mx", "mfs"],  # Literatura Mexico
    122: ["hu", "hu", "hsh"],  # Dictionary Hungary
    123: ["hu", "hu", "hsh"],  # Literature Hungary
    124: ["fr", "fr", "fsl"],  # Literature France
    125: ["en", "gb", "bfi"],  # Literature Great Britain
    126: ["ar", "tn", "tse"],  # Littérature Tunisien
    127: ["mt", "mt", "mdl"],  # Literature Malta
    128: ["mw", "mw", "lws"],  # Dictionary Malawi
    129: ["gn", "py", "pys"],  # Diccionario Paraguay
    130: ["uk", "ua", "ukl"],  # Dictionary Ukraine
    131: ["is", "is", "icl"],  # Ordabók IS
    132: ["ro", "ro", "rms"],  # Dictionary Romania
    133: ["ne", "np", "nsp"],  # Dictionary Nepal
    134: ["bg", "bg", "bqn"],  # Dictionary Bulgaria
    135: ["es", "cl", "csg"],  # Diccionario Chile
    136: ["es", "ec", "ecs"],  # Diccionario Ecuador
    137: ["es", "sv", "esn"],  # Diccionario El Salvador
    138: ["ro", "ro", "rms"],  # Literature Romania
    139: ["ro", "ro", "rms"],  # Encyclopedia Romania
    140: ["fr", "ca", "fcs"],  # Encyclopédie Quebec
    141: ["ru", "ru", "rsl"],  # Literature Russia
    142: ["ru", "ru", "rsl"],  # Encyclopedia Russia
    143: ["es", "uy", "ugy"],  # Diccionario Uruguay
    144: ["es", "uy", "ugy"],  # Literatura Uruguay
    145: ["es", "ar", "aed"],  # Literatura Argentina
    146: ["es", "ar", "aed"],  # Enciclopedia Argentina
    147: ["mt", "mt", "mdl"],  # Literature Malta Archive
    148: ["sl", "sl", "ysl"],  # Besedila Slovenia
    149: ["sl", "sl", "ysl"],  # Enciklopedija Slovenia
    150: ["", "", ""],  # Anthropology Book Project
    151: ["en", "us", "ase"],  # ASL Bible Books NLT
    152: ["en", "us", "ase"],  # ASL Bible Books Shores Deaf Church
    153: ["vn", "vn", "haf"],  # Dictionary Vietnam
}

CACHE_BUSTER = str(datetime.today().date())

class SignBank(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for SignBank dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    BUILDER_CONFIGS = [SignDatasetConfig(name="default", include_video=False)]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {
            "puddle": tf.int32,
            "id": tfds.features.Text(),
            "assumed_spoken_language_code": tfds.features.Text(),
            "sign_language_code": tfds.features.Text(),
            "country_code": tfds.features.Text(),
            "created_date": tfds.features.Text(),
            "modified_date": tfds.features.Text(),
            "sign_writing": tfds.features.Sequence(tfds.features.Text()),
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
        dataset_warning(self)

        index = dl_manager.download("http://signbank.org/signpuddle2.0/data/spml/")
        regex = r"\"sgn[\d]+.spml\""
        with open(index, "r", encoding="utf-8") as f:
            matches = [match.group()[1:-1] for match in re.finditer(regex, f.read(), re.MULTILINE)]
            spml_urls = ["http://signbank.org/signpuddle2.0/data/spml/" + match + "?buster=" + CACHE_BUSTER
                         for match in matches if match != "sgn0.spml"]

        spmls = dl_manager.download(spml_urls)

        return {
            "train": self._generate_examples(spmls),
        }

    def _generate_examples(self, spmls: List[str]):
        """Yields examples."""

        i = 0
        for spml in tqdm(reversed(spmls)):
            tree = ET.parse(spml)
            root = tree.getroot()

            puddle = int(root.attrib["puddle"])

            children = root.iter()
            for child in children:
                if child.tag == "entry":
                    _id = child.attrib["id"]
                    mdt = int(child.attrib["mdt"]) if "mdt" in child.attrib and child.attrib["mdt"] != "" else 0
                    cdt = int(child.attrib["cdt"]) if "cdt" in child.attrib and child.attrib["cdt"] != "" else mdt
                    usr = child.attrib["usr"] if "usr" in child.attrib else ""
                    texts = [c.text for c in child.iter() if c.tag != "src" and c.text is not None and c.text.strip() != ""]

                    signwriting_texts = []
                    spoken_texts = []
                    for text in texts:
                        if is_signwriting(text):
                            signwriting_texts.append(text)
                        else:
                            spoken_texts.append(text)

                    # print(child.attrib)
                    if len(texts) > 0:
                        sample_id = "_".join([str(puddle), _id, str(i)])
                        assumed_spoken_language_code, country_code, sign_language_code = PUDDLES[puddle] if puddle in PUDDLES else ["", "", ""]
                        i += 1
                        yield sample_id, {
                            "puddle": puddle,
                            "id": _id,
                            "assumed_spoken_language_code": assumed_spoken_language_code,
                            "sign_language_code": sign_language_code,
                            "country_code": country_code,
                            "created_date": str(datetime.fromtimestamp(cdt)),
                            "modified_date": str(datetime.fromtimestamp(mdt)),
                            "sign_writing": signwriting_texts,
                            "terms": spoken_texts,
                            "user": usr,
                        }
