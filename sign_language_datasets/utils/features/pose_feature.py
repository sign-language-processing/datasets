import os
from typing import Optional

import numpy as np
import tensorflow as tf
from pose_format import Pose, PoseHeader
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.reader import BufferReader
from tensorflow_datasets.core.features import feature
from tensorflow_datasets.core.utils import type_utils, Json


def read_header(header_path: str):
    with open(header_path, "rb") as f:
        reader = BufferReader(f.read())
        header = PoseHeader.read(reader)
        return header, reader.read_offset


def read_body(pose_bytes: bytes, header: PoseHeader, read_offset: Optional[int] = None):
    reader = BufferReader(pose_bytes)
    if header is None:
        header = PoseHeader.read(reader)
    else:
        reader.read_offset = read_offset

    return NumPyPoseBody.read(header, reader)


# TODO write doc


class PoseFeature(feature.FeatureConnector):
    """`FeatureConnector` for poses.

    During `_generate_examples`, the feature connector accept as input any of:

      * `str`: path to a {bmp,gif,jpeg,png} image (ex: `/path/to/img.png`).
      * `np.array`: 3d `np.uint8` array representing an image.
      * A file object containing the png or jpeg encoded image string (ex:
        `io.BytesIO(encoded_img_bytes)`)

    Output:

      `tf.Tensor` of type `tf.uint8` and shape `[height, width, num_channels]`
      for BMP, JPEG, and PNG images and shape `[num_frames, height, width, 3]` for
      GIF images.

    Example:

      * In the `tfds.core.DatasetInfo` object:

      ```python
      features=features.FeaturesDict({
          "input": features.Image(),
          "target": features.Image(shape=(None, None, 1),
                                     encoding_format="png"),
      })
      ```

      * During generation:

      ```python
      yield {
          "input": "path/to/img.jpg",
          "target": np.ones(shape=(64, 64, 1), dtype=np.uint8),
      }
      ```
    """

    def __init__(self, *, shape=None, header_path: str = None, encoding_format: str = None, stride: int = 1):
        """Construct the connector.

        Args:
          shape: tuple of ints or None, the shape of decoded image.
            For GIF images: (num_frames, height, width, channels=3). num_frames,
              height and width can be None.
            For other images: (height, width, channels). height and width can be
              None. See `tf.image.encode_*` for doc on channels parameter.
            Defaults to (None, None, 3).
          dtype: tf.uint16 or tf.uint8 (default).
            tf.uint16 can be used only with png encoding_format
          encoding_format: "jpeg" or "png". Format to serialize `np.ndarray` images
            on disk. If None, encode images as PNG.
            If image is loaded from {bmg,gif,jpeg,png} file, this parameter is
            ignored, and file original encoding is used.

        Raises:
          ValueError: If the shape is invalid
        """
        # Set and validate values
        self._shape = shape or (None, None, None, 3)
        self._encoding_format = encoding_format or "pose"

        assert int(stride) == stride, "Video fps must be divisible by custom fps, when loading poses"
        self.stride = int(stride)

        if header_path is not None:
            self._header, self._read_offset = read_header(header_path)
        else:
            self._header, self._read_offset = None, None

        self._runner = None

    def get_tensor_info(self):
        # Image is returned as a 3-d uint8 tf.Tensor.
        conf_shape = tuple(list(self._shape)[:3])
        return {
            "data": feature.TensorInfo(shape=self._shape, dtype=tf.float32),
            "conf": feature.TensorInfo(shape=conf_shape, dtype=tf.float32),
            "fps": feature.TensorInfo(shape=(), dtype=tf.int32),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_runner"] = None
        return state

    def encode_body(self, body: NumPyPoseBody):
        if self.stride != 1:
            body = body.slice_step(self.stride)

        return {"data": body.data.data, "conf": body.confidence, "fps": int(body.fps)}

    def encode_example(self, pose_path_or_fobj):
        """Convert the given image into a dict convertible to tf example."""

        if pose_path_or_fobj is None:
            # Create 0 size tensors
            data_shape = list(self._shape)
            data_shape[0] = 0
            conf_shape = tuple(data_shape[:3])
            data_shape = tuple(data_shape)

            pose_body = NumPyPoseBody(data=np.zeros(data_shape), confidence=np.zeros(conf_shape), fps=0)
            return self.encode_body(pose_body)
        elif isinstance(pose_path_or_fobj, Pose):
            return self.encode_body(pose_path_or_fobj.body)
        elif isinstance(pose_path_or_fobj, type_utils.PathLikeCls):
            pose_path_or_fobj = os.fspath(pose_path_or_fobj)
            with tf.io.gfile.GFile(pose_path_or_fobj, "rb") as pose_f:
                encoded_pose = pose_f.read()
        elif isinstance(pose_path_or_fobj, bytes):
            encoded_pose = pose_path_or_fobj
        else:
            encoded_pose = pose_path_or_fobj.read()

        if self._encoding_format == "pose":
            pose_body = read_body(encoded_pose, self._header, self._read_offset)
            return self.encode_body(pose_body)
        else:
            raise Exception("Unknown encoding format '%s'" % self._encoding_format)

    # def decode_example(self, example):
    #     """Reconstruct the image from the tf example."""
    #     print("Decoding", example)
    #     print(example["data"].shape)
    #     print(example["data"])
    #     raise Exception()
    #     img = tf.image.decode_image(
    #         example, channels=self._shape[-1], dtype=self._dtype)
    #     img.set_shape(self._shape)
    #     return img

    @classmethod
    def from_json_content(cls, value: Json) -> "Pose":
        shape = tuple(value["shape"])
        encoding_format = value["encoding_format"]
        return cls(shape=shape, encoding_format=encoding_format)

    def to_json_content(self) -> Json:
        return {
            "shape": list(self._shape),
            "encoding_format": self._encoding_format,
        }
