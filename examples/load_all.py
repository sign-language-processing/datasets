import itertools

from sign_language_datasets.datasets.config import SignDatasetConfig
import tensorflow_datasets as tfds

config = SignDatasetConfig(
    name="pose_holistic_paths2",
    version="3.0.0",
    include_video=False,
    include_pose="holistic",
    process_pose=False
)

# Load the dgs_types dataset with the specified configuration
dgs_types = tfds.load('dgs_types', builder_kwargs=dict(config=config))

for datum in dgs_types["train"].take(10):
    print(datum)