import asyncio
import re
import requests
import os

import tensorflow_datasets as tfds

from tensorflow_datasets.core.download import downloader
from tensorflow_datasets.core import utils
from tensorflow_datasets.core.download import checksums as checksums_lib

from pyppeteer import launch
from typing import Optional, List, Dict, Any


async def login_with_pyppeteer(page, username: str, password: str) -> None:
    """

    :param page:
    :param username:
    :param password:
    :return:
    """

    # <td class="LoginLabel" align="center">

    login_element = (await page.Jx('//td[@class="LoginLabel"]/a'))[0]

    await asyncio.gather(
        page.waitForNavigation(),
        login_element.click()
    )

    # <a>Other Registered Users</a>

    login_element = (await page.Jx('//a[contains(., "Other Registered Users")]'))[0]

    await asyncio.gather(
        page.waitForNavigation(),
        login_element.click()
    )

    input_name = (await page.Jx('//TR[TD[@class = "LoginLabel" and text() = "Name:"]]/TD/INPUT'))[0]
    input_password = (await page.Jx('//TR[TD[@class = "LoginLabel" and contains(text(), "Password:")]]/TD/INPUT'))[0]

    await input_name.focus()
    await page.keyboard.type(username)

    await input_password.focus()
    await page.keyboard.type(password)

    # <input type="submit" class="But" value="Login">
    submit_element = (await page.Jx('//input[@type = "submit"]'))[0]

    await asyncio.gather(
        page.waitForNavigation(),
        submit_element.click()
    )


async def _download_with_pyppeteer(resource_url: str,
                                   download_directory: str,
                                   authenticate: bool = False,
                                   username: Optional[str] = None,
                                   password: Optional[str] = None,
                                   headless: bool = True,
                                   executable_path: Optional[str] = None,
                                   runs_in_main_thread: bool = True) -> None:
    """

    :param resource_url:
    :param download_directory:
    :param authenticate:
    :param username:
    :param password:
    :param headless:
    :param executable_path:
    :param runs_in_main_thread:
    :return:
    """
    if runs_in_main_thread:
        browser = await launch(headless=headless,
                               executablePath=executable_path)
    else:
        browser = await launch(headless=headless,
                               executablePath=executable_path,
                               handleSIGINT=False,
                               handleSIGTERM=False,
                               handleSIGHUP=False)

    page, = await browser.pages()

    # taken from https://github.com/miyakogi/pyppeteer/issues/77#issuecomment-463752650

    cdp = await page.target.createCDPSession()
    await cdp.send('Page.setDownloadBehavior', {'behavior': 'allow', 'downloadPath': download_directory})

    await page.goto(resource_url)

    if authenticate:
        assert None not in (username, password)

        await login_with_pyppeteer(page=page, username=username, password=password)

    # if digital consent pops up, click

    # <a href="javascript:displayViewer();"><span class="But">Continue &gt;&gt;</span></a>

    continue_element = await page.Jx('//a[span[contains(., "Continue")]]')

    if len(continue_element) > 0:
        await asyncio.gather(
            page.waitForNavigation(),
            continue_element[0].click()
        )

    if not headless:
        await asyncio.sleep(10)

    await page.waitForNavigation(options={"waitUntil": "networkidle0"})

    await browser.close()


def download_with_pyppeteer(resource_url: str,
                            download_directory: str,
                            authenticate: bool = False,
                            username: Optional[str] = None,
                            password: Optional[str] = None,
                            headless: bool = True,
                            executable_path: Optional[str] = None,
                            runs_in_main_thread: bool = True) -> None:
    """

    :param resource_url:
    :param download_directory:
    :param authenticate:
    :param username:
    :param password:
    :param headless:
    :param executable_path:
    :param runs_in_main_thread:
    :return:
    """

    if runs_in_main_thread:
        loop_fn = asyncio.get_event_loop
    else:
        loop_fn = asyncio.new_event_loop

    loop_fn().run_until_complete(_download_with_pyppeteer(resource_url=resource_url,
                                                          download_directory=download_directory,
                                                          authenticate=authenticate,
                                                          username=username,
                                                          password=password,
                                                          headless=headless,
                                                          executable_path=executable_path,
                                                          runs_in_main_thread=runs_in_main_thread))


def generate_download_links(username: str,
                            password: str,
                            number_of_records: Optional[int] = None,
                            base_url: str = "http://digital-collections.ucl.ac.uk/") -> List[Dict]:
    """

    :param username:
    :param password:
    :param number_of_records:
    :param base_url:
    :return:
    """

    # Get initial page to generate a user token

    initial_url = base_url + "R&?local_base=BSLCP"
    initial_response = requests.get(initial_url).text

    # Perform the login operation for the retrieved user token

    redirect_url = re.search(r"<a href=\"(.*?)\">Researcher Login<\/a>", initial_response).groups()[0]

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

    # We are logged in, now let's search
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

    search_response = requests.post(search_url, data=post_data).text

    if number_of_records is None:
        number_of_records = int(re.search(r"Records.*?of\s*(\d*)", search_response).groups()[0])

    results_page_response = ""  # all pages are concatenated into a long string

    for i in range(1, number_of_records, 20):  # Page size is 20, no control over that that I have seen
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
        datum = {
            "id": cells[0],
            "record_id": re.search(r">(.*?)<", cells[2]).groups()[0],
            "downloads": {k: v.replace("&amp;", "&") for v, k in downloads}
        }
        index_data.append(datum)

    return index_data


# noinspection PyProtectedMember
@utils.memoize()
def get_downloader_with_pyppeteer(*args: Any, **kwargs: Any) -> '_DownloaderWithPyppeteer':
    return _DownloaderWithPyppeteer(*args, **kwargs)


# noinspection PyProtectedMember
class _DownloaderWithPyppeteer(downloader._Downloader):

    def __init__(self, username: str, password: str, **kwargs):
        super(_DownloaderWithPyppeteer, self).__init__(**kwargs)

        self.username = username
        self.password = password

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

        filename = utils.basename_from_url(url)
        out_path = os.path.join(destination_path, filename)

        download_with_pyppeteer(resource_url=url,
                                download_directory=destination_path,
                                authenticate=False,
                                headless=True,
                                runs_in_main_thread=False)

        url_info = checksums_lib.compute_url_info(
            out_path, checksum_cls=self._checksumer_cls)
        self._pbar_dl_size.update_total(url_info.size)
        self._pbar_dl_size.update(url_info.size)
        self._pbar_url.update(1)
        return downloader.DownloadResult(path=utils.as_path(out_path), url_info=url_info)


class DownloadManagerWithPyppeteer(tfds.download.DownloadManager):

    def __init__(self, *, username: str, password: str, **kwargs):
        super().__init__(**kwargs)

        self.username = username
        self.password = password

        self.__downloader = None

    @property
    def _downloader(self):
        if self.__downloader is None:
            self.__downloader = get_downloader_with_pyppeteer(username=self.username, password=self.password)
        return self.__downloader


if __name__ == "__main__":
    BSLCP_USERNAME = os.environ["BSLCP_USERNAME"]
    BSLCP_PASSWORD = os.environ["BSLCP_PASSWORD"]

    num_example_records = 20

    data = generate_download_links(username=BSLCP_USERNAME, password=BSLCP_PASSWORD, number_of_records=num_example_records)

    num_records_found = 0

    for datum in data:
        # look for a record with an ELAN file
        if "EAF file" not in datum["downloads"].keys():
            continue

        num_records_found += 1
        print()
        print(datum)
        resource_url_elan = datum["downloads"]["EAF file"]

        executable_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        download_directory = "/Users/mathiasmuller/Desktop/bslcp_trial"

        download_with_pyppeteer(resource_url=resource_url_elan,
                                download_directory=download_directory,
                                authenticate=False,
                                headless=False,
                                executable_path=executable_path)
