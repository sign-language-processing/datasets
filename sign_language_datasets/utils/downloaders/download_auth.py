import requests
import shutil
import os
import tensorflow as tf

import tensorflow_datasets as tfds

from tensorflow_datasets.core.download import downloader
from tensorflow_datasets.core import units
from tensorflow_datasets.core import utils
from tensorflow_datasets.core.download import checksums as checksums_lib

from typing import Optional, Any


def download_with_auth(url: str, username: str, password: str, decode: bool = True):
    """
    Download with basic HTTP authentication.

    :param url:
    :param username:
    :param password:
    :param decode:
    :return:
    """
    response = requests.get(url, auth=(username, password))

    if decode:
        return response.content.decode("utf-8")
    else:
        return response.content


def download_tar_gz_to_file_with_auth(url: str,
                                      filepath: str,
                                      username: str,
                                      password: str,
                                      unpack: bool = False,
                                      unpack_path: Optional[str] = None):
    """

    :param url:
    :param filepath:
    :param username:
    :param password:
    :param unpack:
    :param unpack_path:
    :return:
    """
    response = requests.get(url, auth=(username, password), stream=True)

    with open(filepath, 'wb') as outfile:
        outfile.write(response.raw.read())

    if unpack:
        if unpack_path is None:
            assert ".tar.gz" in filepath, "'tar.gz' not found in filepath. Specify an explicit 'unpack_path'."
            unpack_path = filepath.replace(".tar.gz", "")

        shutil.unpack_archive(filepath, unpack_path)


# noinspection PyProtectedMember
@utils.memoize()
def get_downloader_with_auth(*args: Any, **kwargs: Any) -> '_DownloaderWithAuth':
    return _DownloaderWithAuth(*args, **kwargs)


# noinspection PyProtectedMember
class _DownloaderWithAuth(downloader._Downloader):

    def __init__(self, username: str, password: str, **kwargs):
        super(_DownloaderWithAuth, self).__init__(**kwargs)

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
        try:
            # If url is on a filesystem that gfile understands, use copy. Otherwise,
            # use requests (http) or urllib (ftp).
            if not url.startswith('http'):
                return self._sync_file_copy(url, destination_path)
        except tf.errors.UnimplementedError:
            pass

        with downloader._open_url(url, verify=verify, auth=(self.username, self.password)) as (response, iter_content):
            fname = downloader._get_filename(response)
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


class DownloadManagerWithAuth(tfds.download.DownloadManager):

    def __init__(self, *, username: str, password: str, **kwargs):
        super().__init__(**kwargs)

        self.username = username
        self.password = password

        self.__downloader = None

    @property
    def _downloader(self):
        if self.__downloader is None:
            self.__downloader = get_downloader_with_auth(username=self.username, password=self.password)
        return self.__downloader
