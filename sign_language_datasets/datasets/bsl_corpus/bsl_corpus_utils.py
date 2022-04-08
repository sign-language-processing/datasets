import re
import io
import contextlib
import requests
import os
import pympi

import tensorflow_datasets as tfds

import tensorflow as tf
from tensorflow_datasets.core.download import downloader
from tensorflow_datasets.core import utils
from tensorflow_datasets.core import units
from tensorflow_datasets.core.download import checksums as checksums_lib

from requests.adapters import HTTPAdapter, Retry
from typing import Optional, List, Dict, Any, Tuple, Iterator, Iterable


def login_with_credentials(username: str,
                           password: str,
                           base_url: str = "http://digital-collections.ucl.ac.uk/") -> str:
    """

    :param username:
    :param password:
    :param base_url:
    :return:
    """
    # Get initial page to generate a user token

    initial_url = base_url + "R&?local_base=BSLCP"
    initial_response = requests.get(initial_url)

    downloader._assert_status(initial_response)
    initial_response = initial_response.text

    # Perform the login operation for the retrieved user token

    redirect_url = re.search(r"<a href=\"(.*?)\">Researcher Login</a>", initial_response).groups()[0]

    post_data = {
        "func": "login",
        "calling_systemv": "DIGITOOL",
        "selfreg": "",
        "bor_id": username,
        "bor_verification": password,
        "institute": "",
        "url": redirect_url,
    }

    login_response = requests.post(base_url + "pds", data=post_data).text

    login_redirect_response = login_response
    for i in range(2):
        login_redirect_uri = re.search(r"<body onload = \"location = '/(.*?)'", login_redirect_response).groups()[0]
        login_redirect_response = requests.get(base_url + login_redirect_uri, allow_redirects=True).text

    user_token = re.search(r"URL=.*?/R/(.*?)-", login_redirect_response).groups()[0]

    return user_token


def get_search_response(user_token: str,
                        base_url: str = "http://digital-collections.ucl.ac.uk/") -> Tuple[str, str]:
    """

    :param user_token:
    :param base_url:
    :return:
    """
    # The way this works is we have a TOKEN and a request identifier.
    # Seems like they store both, and show the relevant info for the identifier.

    search_url = base_url + 'R/' + user_token + '-00001'
    post_data = {
        "func": "search-advanced-go",
        "file_format_code": "WEX",
        "local_base": "BSLCP",
        "mode": "1",
        "find_code1": "WRD",
        "request1": "bslcp",
        "find_operator": "AND",
        "find_code2": "WRD",
        "request2": "",
        "find_operator2": "AND",
        "find_code3": "WRD",
        "request3": "",
        "media_type": "ALL",
        "selected_otype": "",
        "selected_tag": "0"
    }

    search_response_text = requests.post(search_url, data=post_data).text

    return search_url, search_response_text


def generate_download_links(username: str,
                            password: str,
                            number_of_records: Optional[int] = None,
                            base_url: str = "http://digital-collections.ucl.ac.uk/",
                            renew_user_token_every_n_pages: int = 5) -> Iterator[List[Dict]]:
    """
    Yields 20 search results at a time

    :param username:
    :param password:
    :param number_of_records:
    :param base_url:
    :param renew_user_token_every_n_pages:
    :return:
    """
    if number_of_records is None:
        # do an initial search to find the total number of records
        user_token = login_with_credentials(username=username, password=password, base_url=base_url)

        # We are logged in, now let's search
        _, search_response_text = get_search_response(user_token=user_token, base_url=base_url)

        number_of_records = int(re.search(r"Records.*?of\s*(\d*)", search_response_text).groups()[0])

    for i in range(1, number_of_records, 20):  # Page size is 20, no control over that that I have seen

        if i == 1 or i % renew_user_token_every_n_pages == 0:
            # generate a new user token for this result page
            user_token = login_with_credentials(username=username, password=password, base_url=base_url)

            # search again with new user token
            search_url, _ = get_search_response(user_token=user_token, base_url=base_url)

        results_page_response = ""  # all pages are concatenated into a long string

        results_page_url = f"{search_url}?func=results-jump-page&set_entry={i}&result_format=001"
        results_page_response += requests.get(results_page_url).text

        index_data = []

        last_index = 0
        while True:
            try:
                start_section = results_page_response.index("<!-- START_SECTION --><!-- body-line -->", last_index)
            except ValueError:  # substring not found
                break
            end_section = results_page_response.index("<!-- END_SECTION --><!-- body-line -->", last_index)
            last_index = end_section + 1

            cells = re.findall(r"<TD.*?>(.*?)</TD>", results_page_response[start_section:end_section])
            downloads = re.findall(r"<A.*?open_window_delivery\(\"(.*?)\".*?ALT=\"(.*?)\"", cells[3])

            download_dict = {"Quicktime movie": [], "EAF file": [], "MP4 file": [], "MPEG file": [], "Unknown": []}

            for url, file_type in downloads:
                download_dict[file_type].append(url.replace("&amp;", "&"))

            record_id = re.search(r">(.*?)<", cells[2]).groups()[0]

            datum = {
                "id": cells[0],
                "record_id": record_id,
                "downloads": download_dict
            }
            index_data.append(datum)

        yield index_data


def get_responses_from_container_url(container_url: str,
                                     base_url: str,
                                     max_retries: int = 3) -> Tuple[requests.Response,
                                                                    requests.Response,
                                                                    str]:
    """

    :param container_url:
    :param base_url:
    :param max_retries:
    :return:
    """
    session = requests.Session()

    retry = Retry(connect=max_retries, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    file_container_response = session.get(container_url)
    downloader._assert_status(file_container_response)

    file_container_response_text = file_container_response.text

    cookie = file_container_response.headers['Set-Cookie'].split(';')[0]

    file_container_frame_src = re.search(r'<FRAME SRC=\"/(.*?)\"', file_container_response_text).groups()[0]

    file_container_frame_response = session.get(base_url + file_container_frame_src,
                                                headers={'Cookie': cookie})
    downloader._assert_status(file_container_frame_response)
    file_container_frame_response_text = file_container_frame_response.text

    metadata_url, consent_url = re.search(r'setLabelMetadataStream\(.*, \"(.*?)\", .*, .*, .*, \"(.*?)\"',
                                          file_container_frame_response_text).groups()

    metadata_response = requests.get(metadata_url)
    downloader._assert_status(metadata_response)

    # this returns either a consent reponse or directly the desired file content
    consent_response_or_file_response = session.get(consent_url, stream=True, headers={'Cookie': cookie})
    downloader._assert_status(consent_response_or_file_response)

    return metadata_response, consent_response_or_file_response, cookie


def get_metadata_from_response(metadata_response: requests.Response) -> dict:
    """

    :param metadata_response:
    :return:
    """
    metadata_text = metadata_response.text
    metadata_results = re.findall(r'<td>(.*?):</td>\s*<td>(.*?)<', metadata_text)
    metadata = {k: v for k, v in metadata_results}

    return metadata


@contextlib.contextmanager
def _stream_file_from_container_url(container_url: str,
                                    base_url: str,
                                    max_retries: int = 3,
                                    **kwargs: Any) -> Iterator[Tuple[downloader.Response,
                                                                     Dict,
                                                                     Iterable[bytes]]]:
    """

    :param container_url:
    :param base_url:
    :param max_retries:
    :param kwargs:
    :return:
    """
    metadata_response, main_response, cookie = get_responses_from_container_url(container_url=container_url,
                                                                                base_url=base_url,
                                                                                max_retries=max_retries)

    metadata = get_metadata_from_response(metadata_response)

    main_response_text = main_response.text

    # search for copyrights uri in response
    copyrights_uri_search_results = re.search(r'copyrights_pid.src = \"/(.*?)\"',main_response_text)

    if copyrights_uri_search_results is None:
        # assume consent form did not appear
        yield (main_response, metadata, main_response.iter_content(chunk_size=io.DEFAULT_BUFFER_SIZE))
    else:
        # assume consent form did appear
        with requests.Session() as session:
            copyrights_uri = copyrights_uri_search_results.groups()[0]
            copyright_response = session.get(base_url + copyrights_uri, headers={'Cookie': cookie}, **kwargs)
            downloader._assert_status(copyright_response)

            download_url = re.search(r'window.location.href=\'(.*?)\'', main_response_text).groups()[0]

            with session.get(download_url, stream=True, headers={'Cookie': cookie}, **kwargs) as download_response:
                downloader._assert_status(download_response)

                yield (download_response, metadata, download_response.iter_content(chunk_size=io.DEFAULT_BUFFER_SIZE))


def _get_file_name_bsl_corpus(response: downloader.Response, metadata: Dict) -> str:
    """

    :param response:
    :param metadata:
    :return:
    """
    response_file_name = re.search(r'filename=%22(.*?)%22', response.headers['Content-Disposition']).groups()[0]

    # return the response filename because in many cases the metadata filename is incorrect:
    # a) Identifier is not really the file name, especially if there are several ELAN files for a single identifier
    # b) instead of the identifier what is stored is the date, while in the "date" field some other information
    #    is stored, such as "Mini-DVD". Example: M20c

    return response_file_name


# noinspection PyProtectedMember
@utils.memoize()
def get_bsl_corpus_downloader(*args: Any, **kwargs: Any) -> '_BslCorpusDownloader':
    return _BslCorpusDownloader(*args, **kwargs)


# noinspection PyProtectedMember
class _BslCorpusDownloader(downloader._Downloader):

    def __init__(self,
                 username: str,
                 password: str,
                 max_retries: int = 3,
                 base_url: str = "http://digital-collections.ucl.ac.uk/",
                 **kwargs):
        super(_BslCorpusDownloader, self).__init__(**kwargs)

        self.username = username
        self.password = password
        self.max_retries = max_retries
        self.base_url = base_url

    def _sync_download(self,
                       url: str,
                       destination_path: str,
                       verify: bool = True) -> downloader.DownloadResult:
        """Synchronous version of `download` method.
        To download through a proxy, the `HTTP_PROXY`, `HTTPS_PROXY`,
        `REQUESTS_CA_BUNDLE`,... environment variables can be exported, as
        described in:
        https://requests.readthedocs.io/en/master/user/advanced/#proxies
        Args:
          url: url to download
          destination_path: path where to write it
          verify: whether to verify ssl certificates
        Returns:
          None
        Raises:
          DownloadError: when download fails.
        """
        # files can have unknown names as a fallback

        with _stream_file_from_container_url(container_url=url,
                                             base_url=self.base_url,
                                             max_retries=self.max_retries,
                                             verify=verify) as (response, metadata, iter_content):
            # fname = downloader._get_filename(response)
            fname = _get_file_name_bsl_corpus(response, metadata)
            path = os.path.join(destination_path, fname)
            size = 0

            # Initialize the download size progress bar
            size_mb = 0
            unit_mb = units.MiB
            total_size = int(response.headers.get('Content-length', 0)) // unit_mb
            self._pbar_dl_size.update_total(total_size)
            with tf.io.gfile.GFile(path, 'wb') as file_:
                checksum = self._checksumer_cls()
                for block in iter_content:
                    size += len(block)
                    checksum.update(block)
                    file_.write(block)

                    # Update the download size progress bar
                    size_mb += len(block)
                    if size_mb > unit_mb:
                        self._pbar_dl_size.update(size_mb // unit_mb)
                        size_mb %= unit_mb
        self._pbar_url.update(1)
        return downloader.DownloadResult(
            path=utils.as_path(path),
            url_info=checksums_lib.UrlInfo(
                checksum=checksum.hexdigest(),
                size=utils.Size(size),
                filename=fname,
            ),
        )


class DownloadManagerBslCorpus(tfds.download.DownloadManager):

    def __init__(self, *, username: str,
                 password: str,
                 max_retries: int = 3,
                 **kwargs):
        super().__init__(**kwargs)

        self.username = username
        self.password = password
        self.max_retries = max_retries

        self.__downloader = None

    @property
    def _downloader(self):
        if self.__downloader is None:
            self.__downloader = get_bsl_corpus_downloader(username=self.username,
                                                          password=self.password,
                                                          max_retries=self.max_retries)
        return self.__downloader


def get_elan_sentences_bsl_corpus(elan_path: str) -> Iterator:
    """
    Third release notes and annotation conventions:
    https://bslcorpusproject.org/wp-content/uploads/BSLCorpus_AnnotationConventions_v3.0_-March2017.pdf
    https://bslcorpusproject.org/wp-content/uploads/Notes-to-the-3rd-release-of-BSL-Corpus-annotations.pdf

    Names of tiers:

    - "LH-IDgloss" for left hand glosses
    - "RH-IDgloss" for right hand glosses
    - "Free Translation" for English translation

    :param elan_path:
    :return:
    """

    eaf = pympi.Elan.Eaf(elan_path)  # TODO add "suppress_version_warning=True" when pympi 1.7 is released

    timeslots = eaf.timeslots

    english_tier_name = "Free Translation"
    if english_tier_name not in eaf.tiers:
        return

    # tiers is defined as follows (http://dopefishh.github.io/pympi/Elan.html):
    #   {tier_name -> (aligned_annotations, reference_annotations, attributes, ordinal)}
    # aligned_annotations:
    #   [{id -> (begin_ts, end_ts, value, svg_ref)}]
    # reference_annotations:
    #   [{id -> (reference, value, previous, svg_ref)}]
    #
    # - "ts" means timeslot, which references a time value in miliseconds
    # - "value" is the actual annotation content
    # - "svg_ref" is an optional reference to an SVG image that is always None in our files

    english_text = list(eaf.tiers[english_tier_name][0].values())

    # collect all glosses in the entire ELAN file

    all_glosses = []

    for hand in ["R", "L"]:
        hand_tier = hand + "H-IDgloss"
        if hand_tier not in eaf.tiers:
            continue

        glosses = {}

        for gloss_id, (start, end, value, _) in eaf.tiers[hand_tier][0].items():
            glosses[gloss_id] = {"start": timeslots[start],
                                 "end": timeslots[end],
                                 "gloss": value,
                                 "hand": hand}

        all_glosses += list(glosses.values())

    for (start, end, value, _) in english_text:
        sentence = {"start": timeslots[start],
                    "end": timeslots[end],
                    "english": value}

        # Add glosses whose timestamps are within this sentence
        glosses_in_sentence = [item for item in all_glosses if
                               item["start"] >= sentence["start"]
                               and item["end"] <= sentence["end"]]

        sentence["glosses"] = list(sorted(glosses_in_sentence, key=lambda d: d["start"]))

        yield sentence



if __name__ == "__main__":
    # running this module as main writes around 6 *.eaf files to the current directory

    BSLCP_USERNAME = os.environ["BSLCP_USERNAME"]
    BSLCP_PASSWORD = os.environ["BSLCP_PASSWORD"]

    base_url = "http://digital-collections.ucl.ac.uk/"

    num_example_records = 20

    download_links_iterator = generate_download_links(username=BSLCP_USERNAME,
                                                      password=BSLCP_PASSWORD,
                                                      number_of_records=num_example_records)

    for data_per_results_page in download_links_iterator:

        for datum in data_per_results_page:
            # look for a record with an ELAN file
            if "EAF file" not in datum["downloads"].keys():
                continue

            resource_url_elan = datum["downloads"]["EAF file"][0]

            with _stream_file_from_container_url(container_url=resource_url_elan,
                                                 base_url=base_url,
                                                 max_retries=5) as (response, metadata, iter_content):

                file_name = _get_file_name_bsl_corpus(response, metadata)
                print("Found %s" % file_name)
                with open(file_name, "wb") as outfile:
                    for block in iter_content:
                        outfile.write(block)
