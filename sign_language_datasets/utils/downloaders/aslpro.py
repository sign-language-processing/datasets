import urllib.request
from urllib.error import ContentTooShortError

_ASLPRO_HEADERS = [
    ("User-Agent", "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7"),
    ("Referer", "http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi"),
]


def download_aslpro(url: str, dst_path: str):
    # ASL Pro videos are in swf format
    opener = urllib.request.build_opener()
    opener.addheaders = _ASLPRO_HEADERS
    urllib.request.install_opener(opener)

    try:
        urllib.request.urlretrieve(url, dst_path)
    except ContentTooShortError:
        pass
