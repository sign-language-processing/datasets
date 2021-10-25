"""how2sign dataset."""

import tensorflow_datasets as tfds

from . import chicagofswild


class ChicagoFSWildTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for how2sign dataset."""
    # TODO(how2sign):
    DATASET_CLASS = chicagofswild.ChicagoFSWild
    SPLITS = {
        'train': 3,  # Number of fake train example
        'test': 1,  # Number of fake test example
    }

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
    # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
    tfds.testing.test_main()
