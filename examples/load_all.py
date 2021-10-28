import itertools

import tensorflow_datasets as tfds
from dotenv import load_dotenv

# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets

# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets.dgs_corpus
from sign_language_datasets.datasets.config import SignDatasetConfig

load_dotenv()

# config = SignDatasetConfig(name="only-annotations", version="3.0.0", include_video=False)
# rwth_phoenix2014_t = tfds.load(name="rwth_phoenix2014_t", builder_kwargs=dict(config=config))

# config = SignDatasetConfig(name="256x256:10", include_video=True, fps=10, resolution=(256, 256))

# aslg_pc12 = tfds.load('aslg_pc12')
#
# rwth_phoenix2014_t = tfds.load('rwth_phoenix2014_t', builder_kwargs=dict(config=config))

# wlasl = tfds.load('wlasl', builder_kwargs=dict(config=config))
#
# autsl = tfds.load('autsl', builder_kwargs=dict(
#     config=SignDatasetConfig(name="test", include_video=False, include_pose="holistic"),
# ))

# dgs_config = SignDatasetConfig(name="holistic-pose", include_video=True, process_video=False, include_pose="holistic")
# dgs_corpus = tfds.load('dgs_corpus', builder_kwargs=dict(config=dgs_config))

# print([d["p.ose"]["data"].shape for d in iter(autsl["train"])])
# print([d["video"].shape for d in iter(autsl["train"])])

chicago_fs_wild = tfds.load('chicago_fs_wild', builder_kwargs=dict(
    config=SignDatasetConfig(name="test", include_video=True, resolution=(100, 100)),
))

for datum in itertools.islice(chicago_fs_wild["train"], 0, 10):
    print(datum["video"].shape)