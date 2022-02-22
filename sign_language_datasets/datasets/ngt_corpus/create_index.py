"""Helper file to crawl TheLanguageArchive and create an up-to-date index of the ngt dataset"""

import json
import re
import os
import sys
import urllib.request

from tqdm import tqdm
from typing import Tuple, Optional


"""
Notes about files and filenames:

The corpus consists of the following files:
- ELAN (*.eaf)
- videos (*.mpg)
- audio (*.wav)

Not all of these file types are present for all corpus items.

All filenames start with an example ID, examples:
- CNGT0100_r3.eaf
- CNGT0100_S002_b.mpg
- CNGT0100.wav

## ELAN Files

The filenames of ELAN files includes the release version, for instance "r3" for the third release. The ELAN XML
also contains references to all video files that are linked to these particular annotations. Example:

<ANNOTATION_DOCUMENT>
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
        <MEDIA_DESCRIPTOR MEDIA_URL="file:///F:/CNGT MPEG-1 body/CNGT0100_S002_b.mpg" MIME_TYPE="video/mpeg" RELATIVE_MEDIA_URL="../../../../CNGT MPEG-1 body/CNGT0092_S002_b.mpg"/>
        <MEDIA_DESCRIPTOR MEDIA_URL="file:///F:/CNGT MPEG-1 body/CNGT0100_S001_b.mpg" MIME_TYPE="video/mpeg" RELATIVE_MEDIA_URL="../../../../CNGT MPEG-1 body/CNGT0092_S001_b.mpg"/>

## Videos

Filenames of videos have the following structure:

[example ID]_[speaker ID]_[view identifier].mpg

Example: CNGT0100_S002_b.mpg

The speaker ID is a unique identifier for a specific person, unlike in the public DGS, where identifiers "a" and "b"
refer to different people depending on the corpus example.

There are 4 types of camera views:
- "f": face of speaker, frontal view, resolution: 352 x 288
- "b": body of speaker, including the face, frontal view, resolution: 352 x 288
- "t": top, the speaker filmed from above, resolution: 352 x 288
- no view identifier and no speaker (i.e. "CNGT0100.mpg" for example): wider-angle video that
  shows the entire scene, resolution: 704 x 288

As far as I can see, "f" and "b" are present for all corpus examples, and for all speakers. "t" and the entire scene
are missing sometimes.

Framerate of videos: 25 fps

## Restricted files

Some files are restricted and cannot be found by the crawler below. This does not mean that all files are restricted,
in some cases the ELAN file is public (albeit empty) but the videos are restricted.

Example: https://archive.mpi.nl/tla/islandora/object/tla%3A1839_00_0000_0000_0009_2D68_9

ELAN files for which the videos are not public are marked with "NP" in the file name, example: "CNGT0020_r3_NP.eaf".

The opposite case is also possible: sometimes only a video of the combined or "t" view is available, while everything else
including the ELAN file is restricted.

Example: CNGT0314 here: https://archive.mpi.nl/tla/islandora/object/tla%3A1839_00_0000_0000_0009_0016_1

List of all such exceptions currently: {'CNGT0025', 'CNGT0314', 'CNGT1542', 'CNGT0020', 'CNGT0876'}
"""


base_path = "https://archive.mpi.nl"


def get_sub_pages(html):
    return [x for x in re.findall('<strong class=\"field-content\"><a href=\"(.*)\">.*</a></strong>', html)]


def create_flat_download_dict() -> dict:

    # Iterate over pages
    next_page = "/tla/islandora/object/tla%3A1839_00_0000_0000_0004_F3D5_A"
    sub_pages = []
    while next_page is not None:
        with urllib.request.urlopen(base_path + next_page) as response:
            html = response.read().decode("utf-8")

            next_search = re.search('<a href=\"(.*?)\">next</a>', html)
            next_page = next_search.group(1) if next_search else None

            sub_pages += get_sub_pages(html)

    # Iterate over sub pages
    download_pages = []
    for page in tqdm(sub_pages):
        print(base_path + page)
        with urllib.request.urlopen(base_path + page) as response:
            html = response.read().decode("utf-8")
            download_pages += get_sub_pages(html)

    # Iterate over download pages (creates a flat list of files)

    flat_download_dict = {}

    for page in tqdm(download_pages):
        print(base_path + page)
        with urllib.request.urlopen(base_path + page) as response:
            html = response.read().decode("utf-8")

            files = re.findall(
                'title=\"view item\">(.*)</a></div>\n<div class=\"flat-compound-buttons\">\n<div class=\"flat-compound-download\"><a href=\"(.*)\" class=\"flat-compound-download\" title=\"download file\"></a>',
                html)
            for file in files:
                flat_download_dict[file[0]] = base_path + file[1]

    return flat_download_dict


def get_id_from_filename(filename: str) -> str:
    """

    :param filename:
    :return:
    """
    if "_" not in filename:
        # assume combined view
        example_id = filename.split(".")[0]
    else:
        example_id = filename.split("_")[0]
    assert example_id.startswith("CNGT")
    return example_id


def get_info_from_mpg_filename(filename: str) -> Tuple[Optional[str], str]:
    """

    :param filename:
    :return:
    """
    if "S" not in filename:
        # assume combined view
        speaker_id = None
    else:
        speaker_id = filename.split("_")[1].split(".")[0]
        assert speaker_id.startswith("S"), "Speaker ID of filename '%s' does not start with S: '%s'" \
                                           % (filename, speaker_id)

    if "_b" in filename:
        view = "b"
    elif "_f" in filename:
        view = "f"
    elif "_t" in filename:
        view = "t"
    else:
        # assume combined view
        view = "c"

    return speaker_id, view


def create_structured_download_dict(force_rebuild: bool = False,
                                    flat_json_path: str = "ngt_flat.json",
                                    structured_json_path: str = "ngt.json"):
    """

    :param force_rebuild:
    :param flat_json_path:
    :param structured_json_path:
    :return:
    """
    # open or create flat download dict

    if os.path.exists(flat_json_path) and not force_rebuild:
        print("Opening existing flat download dict: '%s'" % flat_json_path)
        with open(flat_json_path, "r") as infile:
            flat_download_dict = json.load(infile)
    else:
        print("Rebuilding flat download dict: '%s'" % flat_json_path)
        flat_download_dict = create_flat_download_dict()

        with open(flat_json_path, "w") as outfile:
            json.dump(flat_download_dict, outfile)

    # group files by corpus example ID

    structured_download_dict = {}

    for filename, url in flat_download_dict.items():

        if "-" in filename:
            print("Found hyphen in filename: '%s'. Will replace with underscore." % filename)
            filename = filename.replace("-", "_")

        example_id = get_id_from_filename(filename)

        if example_id not in structured_download_dict.keys():
            structured_download_dict[example_id] = {}

        if ".wav" in filename:
            structured_download_dict[example_id]["audio"] = url
        elif ".eaf" in filename:
            structured_download_dict[example_id]["eaf"] = url
        elif ".mpg" in filename:
            speaker_id, view = get_info_from_mpg_filename(filename)

            if view == "c":
                video_id = "video_c"
            else:
                video_id = "video_" + speaker_id + "_" + view

            structured_download_dict[example_id][video_id] = url
        else:
            print("Unexpected filename: %s" % filename)
            sys.exit()

    with open(structured_json_path, "w") as outfile:
        print("Writing structured download dict '%s'." % "ngt.json")
        json.dump(structured_download_dict, outfile)


if __name__ == "__main__":
    create_structured_download_dict(flat_json_path="ngt_flat.json", structured_json_path="ngt.json")
