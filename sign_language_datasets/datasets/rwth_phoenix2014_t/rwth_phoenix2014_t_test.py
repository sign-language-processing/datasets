"""rwth_phoenix_2014_t dataset."""

import tensorflow_datasets as tfds

from . import rwth_phoenix2014_t


class RWTHPhoenix2014TVideosTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for rwth_phoenix_2014_t dataset."""

    DATASET_CLASS = rwth_phoenix2014_t.RWTHPhoenix2014T
    SPLITS = {"train": 1, "validation": 1, "test": 1}
    OVERLAPPING_SPLITS = ["train", "validation", "test"]
    BUILDER_CONFIG_NAMES_TO_TEST = ["videos"]

    DL_EXTRACT_RESULT = ["annotations"]


class RWTHPhoenix2014TPosesTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for rwth_phoenix_2014_t dataset."""

    DATASET_CLASS = rwth_phoenix2014_t.RWTHPhoenix2014T
    SPLITS = {"train": 1, "validation": 1, "test": 1}
    OVERLAPPING_SPLITS = ["train", "validation", "test"]
    BUILDER_CONFIG_NAMES_TO_TEST = ["poses"]

    DL_EXTRACT_RESULT = ["annotations", "poses"]


if __name__ == "__main__":
    tfds.testing.test_main()
