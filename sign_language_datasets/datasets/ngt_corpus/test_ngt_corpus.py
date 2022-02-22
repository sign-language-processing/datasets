"""ngt_corpus dataset tests."""

import itertools

import tensorflow_datasets as tfds

from unittest import TestCase

from ... import datasets
from ...datasets.config import SignDatasetConfig
from ...datasets.ngt_corpus.ngt_corpus_utils import get_elan_sentences_ngt_corpus


class TestNgtCorpus(TestCase):

    def test_ngt_corpus_loader(self):

        config = SignDatasetConfig(name="only-annotations", version="1.0.0", include_video=False)
        ngt = tfds.load(name='ngt_corpus', builder_kwargs={"config": config})

        for datum in itertools.islice(ngt["train"], 0, 10):
            print(datum)


class TestNgtCorpusUtils(TestCase):

    def test_ngt_get_elan_sentences(self):

        config = SignDatasetConfig(name="only-annotations", version="1.0.0", include_video=False)
        ngt = tfds.load(name='ngt_corpus', builder_kwargs={"config": config})

        for datum in itertools.islice(ngt["train"], 0, 50):
            print(datum['id'].numpy().decode('utf-8'))
            elan_path = datum["paths"]["eaf"].numpy().decode('utf-8')

            sentences = get_elan_sentences_ngt_corpus(elan_path)

            for sentence in sentences:
                print(sentence)
