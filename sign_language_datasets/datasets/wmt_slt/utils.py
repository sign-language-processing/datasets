"""
Utils file contains a subset of utils from:
https://github.com/bricksdont/sign-sockeye-baselines/blob/f056c50765bf5f088419591b505b9f967605d483/scripts/preprocessing/convert_and_split_data.py
"""

import os
import datetime
from typing import Dict

import cv2
import tarfile
import tempfile

import numpy as np

from pose_format import Pose, PoseHeader
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderDimensions
from pose_format.utils.openpose_135 import load_openpose_135_directory
from pose_format.utils.openpose import load_frames_directory_dict


def extract_tar_xz_file(filepath: str, target_dir: str):
    """

    :param filepath:
    :param target_dir:
    :return:
    """
    with tarfile.open(filepath) as tar_handle:
        tar_handle.extractall(path=target_dir)


def read_openpose_surrey_format(filepath: str, fps: int, width: int, height: int) -> Pose:
    """
    Read files of the form "focusnews.071.openpose.tar.xz".
    Assumes a 135 keypoint Openpose model.

    :param filepath:
    :param fps:
    :return:
    """
    with tempfile.TemporaryDirectory(prefix="extract_pose_file") as tmpdir_name:
        # extract tar.xz
        extract_tar_xz_file(filepath=filepath, target_dir=tmpdir_name)

        openpose_dir = os.path.join(tmpdir_name, "openpose")

        # load directory
        poses = load_openpose_135_directory(directory=openpose_dir, fps=fps, width=width, height=height)

    return poses


def formatted_holistic_pose(width=1000, height=1000):

    try:
        import mediapipe as mp
        from pose_format.utils.holistic import holistic_components

    except ImportError:
        raise ImportError("Please install mediapipe with: pip install mediapipe")

    mp_holistic = mp.solutions.holistic
    FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))]

    dimensions = PoseHeaderDimensions(width=width, height=height, depth=width)
    header = PoseHeader(version=0.1, dimensions=dimensions, components=holistic_components("XYZC", 10))
    body = NumPyPoseBody(
        fps=10, data=np.zeros(shape=(1, 1, header.total_points(), 3)), confidence=np.zeros(shape=(1, 1, header.total_points()))
    )
    pose = Pose(header, body)
    return pose.get_components(
        ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"], {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS}
    )


def load_mediapipe_directory(directory: str, fps: int, width: int, height: int) -> Pose:
    """

    :param directory:
    :param fps:
    :return:
    """

    frames = load_frames_directory_dict(directory=directory, pattern="(?:^|\D)?(\d+).*?.json")

    def load_mediapipe_frame(frame):
        def load_landmarks(name, num_points: int):
            points = [[float(p) for p in r.split(",")] for r in frame[name]["landmarks"]]
            points = [(ps + [1.0])[:4] for ps in points]  # Add visibility to all points
            if len(points) == 0:
                points = [[0, 0, 0, 0] for _ in range(num_points)]
            return np.array([[x, y, z] for x, y, z, c in points]), np.array([c for x, y, z, c in points])

        face_data, face_confidence = load_landmarks("face_landmarks", 128)
        body_data, body_confidence = load_landmarks("pose_landmarks", 33)
        lh_data, lh_confidence = load_landmarks("left_hand_landmarks", 21)
        rh_data, rh_confidence = load_landmarks("right_hand_landmarks", 21)
        data = np.concatenate([body_data, face_data, lh_data, rh_data])
        conf = np.concatenate([body_confidence, face_confidence, lh_confidence, rh_confidence])
        return data, conf

    def load_mediapipe_frames():
        max_frames = int(max(frames.keys())) + 1
        pose_body_data = np.zeros(shape=(max_frames, 1, 21 + 21 + 33 + 128, 3), dtype=np.float)
        pose_body_conf = np.zeros(shape=(max_frames, 1, 21 + 21 + 33 + 128), dtype=np.float)
        for frame_id, frame in frames.items():
            data, conf = load_mediapipe_frame(frame)
            pose_body_data[frame_id][0] = data
            pose_body_conf[frame_id][0] = conf
        return NumPyPoseBody(data=pose_body_data, confidence=pose_body_conf, fps=fps)

    pose = formatted_holistic_pose(width=width, height=height)

    pose.body = load_mediapipe_frames()

    return pose


def read_mediapipe_surrey_format(filepath: str, fps: int, width: int, height: int) -> Pose:
    """
    Read files of the form "focusnews.103.mediapipe.tar.xz"
    """
    with tempfile.TemporaryDirectory(prefix="extract_pose_file") as tmpdir_name:
        # extract tar.xz
        extract_tar_xz_file(filepath=filepath, target_dir=tmpdir_name)
        poses_dir = os.path.join(tmpdir_name, "poses")
        # load directory
        pose = load_mediapipe_directory(directory=poses_dir, fps=fps, width=width, height=height)
    return pose


def get_video_metadata(filename: str) -> Dict[str, int]:
    """
    Get metadata from mp4 video.

    :param filename:
    :return:
    """
    cap = cv2.VideoCapture(filename)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {"fps": fps, "width": width, "height": height}


def milliseconds_to_frame_index(milliseconds: int, fps: int) -> int:
    """
    :param miliseconds:
    :param fps:
    :return:
    """
    return int(fps * (milliseconds / 1000))


def convert_srt_time_to_frame(srt_time: datetime.timedelta, fps: int) -> int:
    """
    datetime.timedelta(seconds=4, microseconds=71000)

    :param srt_time:
    :param fps:
    :return:
    """
    seconds, microseconds = srt_time.seconds, srt_time.microseconds

    milliseconds = int((seconds * 1000) + (microseconds / 1000))

    return milliseconds_to_frame_index(milliseconds=milliseconds, fps=fps)


def reduce_pose_people(pose: Pose):
    pose.body.data = pose.body.data[:, 0:1, :, :]
    pose.body.confidence = pose.body.confidence[:, 0:1, :]
