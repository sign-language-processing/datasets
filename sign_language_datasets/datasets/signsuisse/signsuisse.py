"""A Swiss Sign Language Lexicon, combines all three Swiss sign languages: the German-Swiss Sign Language (DSGS), the Langue des Signes Française (LSF) and the Lingua Italiana dei Segni (LIS). ."""
import json
import re
import string

import tensorflow_datasets as tfds
from tqdm import tqdm

from ..warning import dataset_warning
from ...datasets import SignDatasetConfig

_DESCRIPTION = """
Ein umfangreiches interaktives Lexikon der drei Gebärdensprachen der Schweiz (Deutschschweizerische Gebärdensprache DSGS, Langue des Signes Française LSF und Lingua Italiana dei Segni LIS). Herausgegeben vom Schweizerischen Gehörlosenbund (SGB-FSS)."""

# TODO(signsuisse): BibTeX citation
_CITATION = """
"""

SITE_URL = "https://signsuisse.sgb-fss.ch"


class SignSuisse(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for signsuisse dataset."""

    MAX_SIMULTANEOUS_DOWNLOADS = 100  # Default is 50, but it is slow
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial crawl.",
    }

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=True),
        SignDatasetConfig(name="annotations", include_video=False),
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
            "exampleVideo": tfds.features.Text(),
        }

        if self._builder_config.include_video and self._builder_config.process_video:
            features["video"] = self._builder_config.video_feature((640, 480))
        else:
            features["video"] = tfds.features.Text()

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage=SITE_URL,
            supervised_keys=None,
            citation=_CITATION,
        )

    def _list_all_lexicon_items(self, dl_manager: tfds.download.DownloadManager):
        try:
            from unidecode import unidecode
        except ImportError:
            raise ImportError("Please install unidecode with: pip install unidecode")

        chars = string.ascii_lowercase + ' '
        # The lexicon does not allow free search. One must search at least two letters.
        next_searches = [c1 + c2 for c1 in chars for c2 in chars if not (c1 == c2 == " ")]

        tfds.disable_progress_bar()

        done_searches = set()
        results_count = {}

        with open('searches.txt', 'w') as f:
            while len(next_searches) > 0:
                print(len(next_searches), "Next searches")
                search_url = SITE_URL + "/index.php?eID=signsuisse_search&sword="
                urls = {l: search_url + l.replace(' ', '%20') for l in next_searches}
                indexes = dl_manager.download(urls)

                next_searches = []

                for search_term, index in tqdm(indexes.items()):
                    f.write(search_term + '\n')

                    with open(index, "r", encoding="utf-8") as f:
                        index = json.load(f)

                    for item in index["items"]:
                        lexicon_items[item["uid"]] = item

                    done_searches.add(search_term)
                    results_count[search_term] = index["count"]

                    # As far as I know, there's no way to paginate, so this is the only way to get all items.
                    # First it searches 728 terms, (15491 items found)
                    # then 6318 more, (18217 items found),
                    # then 5427 more (18221 items found)
                    # then 540 more (18221 items found)
                    # then 27 more (18221 items found)
                    if len(index["items"]) < index["count"]:
                        # It seems like the search system allows for a small edit distance, which makes it not too efficient to search through
                        # Make sure all items have the search term in them
                        have_search = [item for item in index["items"] if search_term in unidecode(item["name"]).lower()]
                        if len(have_search) == len(index["items"]):
                            for l in chars:
                                next_searches.append(search_term + l)
                                should_subset[search_term] = search_term


                next_searches = list(set([s for s in next_searches if s not in done_searches]))

                print("So far", len(lexicon_items), "items found.")

        return lexicon_items.values()

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

        if item["uid"] in ["131220", "131073", "120731", "115584", "121152", "121125", "126042"]:
            # This is a special case where the video is not available.
            return None

        video = SITE_URL + re.search(r"<video id=\"video-main\".*?src=\"(.*?)\"", html).group(1).strip()
        paraphrase_match = re.search(r"Umschreibung</h2> <p>(.*?)</p>", html)
        paraphrase = paraphrase_match.group(1).strip() if paraphrase_match else ""
        definition_match = re.search(r"Definition</h2> <p>(.*?)</p>", html)
        definition = definition_match.group(1).strip() if definition_match else ""
        spoken_language = item["sprache"]
        url_path = re.search(rf"<a href=\"(\/{spoken_language}\/.*?)\"", html).group(1).strip()

        return {
            "id": item["uid"],
            "name": item["name"],
            "category": item["kategorie"],
            "spokenLanguage": spoken_language,
            "signedLanguage": "ch-" + spoken_language,
            "url": SITE_URL + url_path,
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
        item_urls = [SITE_URL + item["link"] for item in lexicon_items]
        # Item URLS are actually too long. We need to shorten them.
        item_urls = [url[:url.find("&tx_issignsuisselexikon_anzeige%5Baction")] for url in item_urls]
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
                datum["video"] = video
                if not self._builder_config.process_video:
                    datum["video"] = str(datum["video"])

        return {"train": self._generate_examples(data)}

    def _generate_examples(self, data):
        for datum in data:
            yield datum["id"], datum



