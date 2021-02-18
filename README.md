# Sign Language Datasets

This repository includes TFDS data loaders for sign language datasets.

## Installation

#### From Source
```bash
pip install git+https://github.com/sign-language-processing/datasets.git
```

#### PyPi
Not available. Need to add automatic publication on push.

## Usage

We demonstrate a loading script for every dataset in [examples/load.ipynb](examples/load.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sign-language-processing/datasets/blob/master/examples/load.ipynb)

Our config includes the option to choose the resolution and fps, for example:

```python
import tensorflow_datasets as tfds
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig

# Loading a dataset with default configuration
aslg_pc12 = tfds.load("aslg_pc12")

# Loading a dataset with custom configuration
config = SignDatasetConfig(name="videos_and_poses256x256:12", 
                           version="3.0.0",          # Specific version
                           include_video=True,       # Download and load dataset videos
                           process_video=True,       # Process videos to tensors, or only save path to video
                           fps=12,                   # Load videos at constant, 12 fps
                           resolution=(256, 256),    # Convert videos to a constant resolution, 256x256
                           include_pose="holistic")  # Download and load Holistic pose estimation
rwth_phoenix2014_t = tfds.load(name='rwth_phoenix2014_t', builder_kwargs=dict(config=config))
```

## Datasets

| Dataset            | Videos | Poses                                                 | Versions |
|--------------------|--------|-------------------------------------------------------|----------|
| aslg_pc12          | N/A    | N/A                                                   | 0.0.1    |
| rwth_phoenix2014_t | Yes    | Holistic                                              | 3.0.0    |
| autsl              | Yes    | Holistic                                              | 1.0.0    |
| dgs_corpus         | Yes    | OpenPose                                              | 3.0.0    |
| wlasl              | [Failed](https://github.com/tensorflow/datasets/issues/2960)   | [OpenPose](https://github.com/gulvarol/bsl1k/issues/4) | None    |
| msasl              |        |                                                       | None     |
| Video-Based CSL    |        |                                                       | None     |
| RVL-SLLL ASL	     |        |                                                       | None     |

## Data Interface

We follow the following interface wherever possible to make it easy to swap datasets.

```python
{
    "id": tfds.features.Text(),
    "signer": tfds.features.Text() | tf.int32,
    "video": tfds.features.Video(shape=(None, HEIGHT, WIDTH, 3)),
    "depth_video": tfds.features.Video(shape=(None, HEIGHT, WIDTH, 1)),
    "fps": tf.int32,
    "pose": {
        "data": tfds.features.Tensor(shape=(None, 1, POINTS, CHANNELS), dtype=tf.float32),
        "conf": tfds.features.Tensor(shape=(None, 1, POINTS), dtype=tf.float32)
    },
    "gloss": tfds.features.Text(),
    "text": tfds.features.Text()
}
```

### Why not Huggingface Datasets?
Huggingface datasets do not work well with videos.
From the lack of native support of the video type, to lack of support of arbitrary tensors.
Furthermore, they currently have memory leaks that prevent from saving even the smallest of video datasets.

### Cite

```bibtex
@misc{moryossef2021datasets, 
    title={Sign Language Datasets},
    author={Moryossef, Amit},
    howpublished={\url{https://github.com/sign-language-processing/datasets}},
    year={2021}
}
```