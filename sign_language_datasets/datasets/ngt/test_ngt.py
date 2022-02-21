"""ngt_corpus dataset tests."""

import itertools

import tensorflow_datasets as tfds

from unittest import TestCase

from ... import datasets
from ...datasets.config import SignDatasetConfig


class TestNgtCorpus(TestCase):

    def test_ngt_corpus_loader(self):

        config = SignDatasetConfig(name="only-annotations", version="1.0.0", include_video=False)
        ngt = tfds.load(name='ngt_corpus', builder_kwargs={"config": config})

        for datum in itertools.islice(ngt["train"], 0, 10):
            print(datum)
