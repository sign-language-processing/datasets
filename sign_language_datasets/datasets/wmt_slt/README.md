# WMT-SLT

Until [merged by `tfds`](https://github.com/tensorflow/datasets/pull/3988), please install our branch, supporting authentication in datasets:
```bash
pip install git+ssh://git@github.com:AmitMY/datasets-1.git
```


```py

import tensorflow_datasets as tfds
# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig

# Populate your access tokens
TOKENS = {
    "zenodo_focusnews_token": "TODO",
    "zenodo_srf_videos_token": "TODO",
    "zenodo_srf_poses_token": "TODO"
}

# Load only the annotations, and include path to video files
config = SignDatasetConfig(name="annotations", version="1.0.0", process_video=False)
wmtslt = tfds.load(name='wmtslt', builder_kwargs={"config": config, **TOKENS})

# Load the annotations and openpose poses
config = SignDatasetConfig(name="openpose", version="1.0.0", process_video=False, include_pose='openpose')
wmtslt = tfds.load(name='wmtslt', builder_kwargs={"config": config, **TOKENS})

# Load the annotations and mediapipe holistic poses
config = SignDatasetConfig(name="holistic", version="1.0.0", process_video=False, include_pose='holistic')
wmtslt = tfds.load(name='wmtslt', builder_kwargs={"config": config, **TOKENS})

# Load the full video frames as a tensor
config = SignDatasetConfig(name="videos", version="1.0.0", process_video=True)
wmtslt = tfds.load(name='wmtslt', builder_kwargs={"config": config, **TOKENS})

decode_str = lambda s: s.numpy().decode('utf-8')
for datum in itertools.islice(wmtslt["train"], 0, 10):
    subtitle = datum['subtitle']
    print(decode_str(subtitle['start']), '-', decode_str(subtitle['end']))
    print(decode_str(subtitle['text']), '\n')

```