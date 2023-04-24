import itertools

import tensorflow_datasets as tfds

# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets.signsuisse
from sign_language_datasets.datasets.config import SignDatasetConfig



def signsuisse():
    config = SignDatasetConfig(name="local-videos", version="1.0.0",
                               include_video=True,
                               include_pose=None,
                               process_video=False)
    dataset = tfds.load(name='sign_suisse', builder_kwargs={"config": config})

    for datum in itertools.islice(dataset["train"], 0, 10):
        print("Concept: ", datum['name'].numpy().decode('utf-8'))
        print("Video: ", datum['video'].numpy().decode('utf-8'))
        print("Example: ", datum['exampleText'].numpy().decode('utf-8'))
        print("Example Video: ", datum['exampleVideo'].numpy().decode('utf-8'))
        print("---")


if __name__ == '__main__':
    signsuisse()
