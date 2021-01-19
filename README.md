# Sign Language Datasets

This repository includes TFDS data loaders for sign language datasets.

### Why not Huggingface Datasets?
Huggingface datasets do not work well with videos.
From the lack of native support of the video type, to lack of support of arbitrary tensors.
Furthermore, they currently have memory leaks that prevent from saving even the smallest of video datasets.

## Datasets

| Dataset            | Videos | Poses                                                 | Ready |
|--------------------|--------|-------------------------------------------------------|-------|
| aslg_pc12          | N/A    | N/A                                                   | Yes   |
| rwth_phoenix2014_t | Yes    | Holistic                                              | Yes   |
| autsl              | Yes    | Holistic                                              | Yes   |
| wlasl              | [Failed](https://github.com/tensorflow/datasets/issues/2960) | [OpenPose](https://github.com/gulvarol/bsl1k/issues/4) | No    |
| msasl              |        |                                                       | No    |

## Usage

See [load_all.py](load_all.py) for loading code for all datasets. 
It includes the option to choose the resolution and fps. Following is a short sample:

```python
aslg_pc12 = tfds.load("aslg_pc12")

config = SignDatasetConfig(name="256x256:12", include_video=True, fps=12, resolution=(256, 256))
rwth_phoenix2014_t = tfds.load("rwth_phoenix2014_t", builder_kwargs=dict(config=config))
```

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