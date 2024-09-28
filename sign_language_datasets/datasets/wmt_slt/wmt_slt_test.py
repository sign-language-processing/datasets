"""sign2mint dataset."""

import tensorflow_datasets as tfds
from . import wmt_slt


class WMTSLTTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for sign2mint dataset."""

    # TODO(wmt_slt):
    DATASET_CLASS = wmt_slt.WMTSLT
    SPLITS = {"train": 3, "test": 1}  # Number of fake train example  # Number of fake test example

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
    # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == "__main__":
    tfds.testing.test_main()
