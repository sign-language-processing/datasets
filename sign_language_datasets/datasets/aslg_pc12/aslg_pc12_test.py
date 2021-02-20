"""aslg_pc12 dataset."""

import tensorflow_datasets as tfds

from . import aslg_pc12


class AslgPc12Test(tfds.testing.DatasetBuilderTestCase):
    """Tests for aslg_pc12 dataset."""

    DATASET_CLASS = aslg_pc12.AslgPc12
    SPLITS = {
        "train": 10,  # Number of fake train example
    }
    DL_DOWNLOAD_RESULT = ["sample-corpus-asl-en.asl", "sample-corpus-asl-en.en"]


if __name__ == "__main__":
    tfds.testing.test_main()
