from typing import List, Optional, Tuple

import cv2
import tensorflow_datasets as tfds


class SignDatasetConfig(tfds.core.BuilderConfig):
    """General BuilderConfig for sign language datasets."""

    def __init__(
        self,
        include_video: bool = True,
        process_video: bool = True,
        include_pose: Optional[str] = None,
        fps: Optional[float] = None,
        resolution: Optional[Tuple[int, int]] = None,
        extra: dict = {},
        **kwargs,
    ):
        """Constructs a RWTHPhoenix2014TConfig.
        Args:
          include_video: bool, whether to include videos in the data.
          process_video: bool, whether to load the videos as tensors or paths
          include_pose: str, what pose data to include.
          fps: float, what pose data to include.
          resolution: (int, int), what resolution of videos to load.
          **kwargs: keyword arguments forwarded to super.
        """
        super(SignDatasetConfig, self).__init__(**kwargs)
        self.include_video = include_video
        self.process_video = process_video
        self.include_pose = include_pose.lower() if include_pose is not None else None

        self.fps = fps
        self.resolution = resolution
        self.extra = extra

    def ffmpeg_args(self):
        args: List[str] = []

        if self.fps is not None:
            args += ["-r", str(self.fps)]

        if self.resolution is not None:
            w, h = self.resolution
            ratio = str(w) + ":" + str(h)
            scale = ratio + ":force_original_aspect_ratio=decrease"
            pad = ratio + ":(ow-iw)/2:(oh-ih)/2"
            args += ["-vf", "scale=" + scale + ",pad=" + pad + ",setsar=1"]

        return args

    def video_feature(self, default_resolution: Tuple[int, int], channels=3):
        w, h = self.resolution if self.resolution is not None else default_resolution

        ffmpeg_extra_args = self.ffmpeg_args()
        return VideoFeature(shape=(None, h, w, channels), ffmpeg_extra_args=ffmpeg_extra_args)


class VideoFeature(tfds.features.Video):
    def encode_example(self, video_or_path_or_fobj):
        """TFDS does not scale images, so this class handles it"""

        # Load images if list of file paths
        if isinstance(video_or_path_or_fobj, list) and isinstance(video_or_path_or_fobj[0], str):
            _, h, w, _ = self.shape
            video_or_path_or_fobj = [cv2.resize(cv2.imread(f), (w, h)) for f in video_or_path_or_fobj]

        return super(VideoFeature, self).encode_example(video_or_path_or_fobj)
