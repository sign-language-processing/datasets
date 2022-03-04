"""
Helper file to crawl The BOBSL Corpus and create an up-to-date index of the dataset.

This script will index all individual files for videos, pose and flow. For subtitles and spottings, it will
only indicate a single *.tar.gz file that contains all files for all dataset examples.

The environment variables BOBSL_USERNAME and BOBSL_PASSWORD must be set when this script is executed:

BOBSL_USERNAME=??? BOBSL_PASSWORD=??? python -m sign_language_datasets.datasets.bobsl.create_index

These credentials can be obtained by signing a license agreement with the data owners:
https://www.robots.ox.ac.uk/~vgg/data/bobsl
"""

import os
import json
import lxml.html

from io import BytesIO
from typing import Dict

from ...utils.downloaders import download_auth


BOBSL_USERNAME = os.environ["BOBSL_USERNAME"]
BOBSL_PASSWORD = os.environ["BOBSL_PASSWORD"]

BASE_URL = "https://thor.robots.ox.ac.uk/~vgg/data/bobsl"


def get(url: str, decode: bool = True):
    """

    :param url:
    :param decode:
    :return:
    """
    return download_auth.download_with_auth(url=url, username=BOBSL_USERNAME, password=BOBSL_PASSWORD, decode=decode)


def parse_sub_index(index: dict, ids_must_exist: bool, base_url_suffix: str, index_key: str,
                    file_extension: str) -> dict:
    """

    :param index:
    :param ids_must_exist:
    :param base_url_suffix:
    :param index_key:
    :param file_extension:
    :return:
    """
    sub_base_url = BASE_URL + "/" + base_url_suffix

    subpage_content = get(sub_base_url, decode=False)
    doc = lxml.html.parse(BytesIO(subpage_content))

    for link_element in doc.xpath("//a[contains(text(), '%s')]" % file_extension):
        example_id = link_element.text.replace(file_extension, "")
        example_url = sub_base_url + "/" + link_element.get("href")

        if ids_must_exist:
            assert example_id in index.keys()
        else:
            assert example_id not in index.keys()
            index[example_id] = {}

        index[example_id][index_key] = example_url

    return index


def create_index() -> Dict[str, Dict[str, str]]:

    index = {}

    # first pass: get example IDs from video subfolder

    index = parse_sub_index(index=index, ids_must_exist=False, base_url_suffix="videos",
                            index_key="video", file_extension=".mp4")

    # second pass: get URLs from pose subfolder

    index = parse_sub_index(index=index, ids_must_exist=True, base_url_suffix="pose",
                            index_key="openpose", file_extension=".tar.gz")

    # third pass: get URLs from flow subfolder

    index = parse_sub_index(index=index, ids_must_exist=True, base_url_suffix="flow",
                            index_key="flow", file_extension=".tar.gz")

    # add subtitles and spottings URLs

    index["subtitles"] = BASE_URL + "/" + "subtitles.tar.gz"
    index["spottings"] = BASE_URL + "/" + "spottings.tar.gz"

    return index


def create_and_write_index(json_path: str) -> None:
    """

    :param json_path:
    :return:
    """
    index = create_index()

    # print some examples as a sanity check
    print("10 Examples from the index:")

    for item_index, kv_tuple in enumerate(index.items()):
        if item_index == 10:
            break
        print(kv_tuple)

    with open(json_path, "w") as outfile:
        print("Writing structured download dict '%s'." % json_path)
        json.dump(index, outfile)


if __name__ == "__main__":
    create_and_write_index(json_path="bobsl.json")
