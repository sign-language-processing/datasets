"""dgs_corpus dataset."""

import json
import gzip
import shutil
import tempfile
import numpy as np
import tensorflow_datasets as tfds

from unittest import TestCase
from typing import List, Dict
from contextlib import contextmanager

from sign_language_datasets.datasets.dgs_corpus import dgs_corpus
from pose_format.pose import Pose


OPENPOSE_TOTAL_KEYPOINTS = 137
OPENPOSE_COMPONENTS_USED = ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]
OPENPOSE_COMPONENTS_UNUSED = ["pose_keypoints_3d", "face_keypoints_3d", "hand_left_keypoints_3d", "hand_right_keypoints_3d"]
OPENPOSE_NUM_POINTS_PER_COMPONENT = {"pose_keypoints_2d": 25,
                                     "face_keypoints_2d": 70,
                                     "hand_left_keypoints_2d": 21,
                                     "hand_right_keypoints_2d": 21}


def _generate_dgs_openpose_data(camera_names: List[str],
                                num_frames: int = 10,
                                num_people: int = 1) -> List[Dict]:
    data = []  # type: List[Dict]

    for camera_name in camera_names:
        camera_dict = {"camera": camera_name}
        camera_dict["id"] = np.random.randint(10000000)
        camera_dict["width"] = 1280
        camera_dict["height"] = 720

        camera_dict["frames"] = {}

        for frame_id in range(num_frames):
            frame_dict = {"version": 1.4, "people": []}

            for _ in range(num_people):
                keypoints_dict = {}
                for component in OPENPOSE_COMPONENTS_USED:
                    num_keypoints = OPENPOSE_NUM_POINTS_PER_COMPONENT[component]

                    # for each keypoint: one value for each dimension, plus confidence
                    num_pose_values = 3 * num_keypoints
                    pose_data = np.random.random_sample((num_pose_values,)).tolist()
                    keypoints_dict[component] = pose_data

                for component in OPENPOSE_COMPONENTS_UNUSED:
                    keypoints_dict[component] = []

                frame_dict["people"].append(keypoints_dict)

            camera_dict["frames"][str(frame_id)] = frame_dict

        data.append(camera_dict)

    return data


def _generate_dgs_openpose_file(filehandle: tempfile.NamedTemporaryFile,
                                camera_names: List[str],
                                num_frames: int = 10,
                                num_people: int = 1) -> None:

    data = _generate_dgs_openpose_data(camera_names, num_frames, num_people)
    json.dump(data, filehandle)
    filehandle.flush()


def _gzip_file(filepath_in: str, filepath_out: str) -> None:
    with open(filepath_in, 'rb') as filehandle_in:
        with gzip.open(filepath_out, 'wb') as filehandle_out:
            shutil.copyfileobj(filehandle_in, filehandle_out)


@contextmanager
def _create_tmp_dgs_openpose_file(camera_names: List[str],
                                  num_frames: int = 10,
                                  num_people: int = 1) -> str:
    with tempfile.NamedTemporaryFile(mode="w+") as filehandle:
        _generate_dgs_openpose_file(filehandle, camera_names, num_frames, num_people)

        filepath_zipped = filehandle.name + ".gz"
        _gzip_file(filehandle.name, filepath_zipped)

        yield filepath_zipped


class TestDgsCorpusAuxiliaryFunctions(TestCase):

    def test_convert_dgs_dict_to_openpose_frames(self):

        input_dict = {"7": {"people": [1, 2, 3]}, "8": {"people": [4, 5, 6]}}

        expected_output = {7: {"people": [1, 2, 3], "frame_id": 7},
                           8: {"people": [4, 5, 6], "frame_id": 8}}

        actual_output = dgs_corpus.convert_dgs_dict_to_openpose_frames(input_dict)

        self.assertDictEqual(actual_output, expected_output)

    def test_get_poses_return_type(self):

        camera_names_in_mock_data = ["a", "b", "c"]
        num_frames_in_mock_data = 10
        num_people_in_mock_data = 1

        people_to_extract = {"a", "b"}

        with _create_tmp_dgs_openpose_file(camera_names=camera_names_in_mock_data,
                                           num_frames=num_frames_in_mock_data,
                                           num_people=num_people_in_mock_data) as filepath:
            poses = dgs_corpus.get_openpose(openpose_path=filepath, fps=50, people=people_to_extract)

            for pose in poses.values():
                self.assertIsInstance(pose, Pose)

    def test_get_poses_subset_of_camera_names(self):

        camera_names_in_mock_data = ["a2", "b1", "c5"]
        num_frames_in_mock_data = 10
        num_people_in_mock_data = 1

        people_to_extract = {"a", "b"}

        with _create_tmp_dgs_openpose_file(camera_names=camera_names_in_mock_data,
                                           num_frames=num_frames_in_mock_data,
                                           num_people=num_people_in_mock_data) as filepath:
            poses = dgs_corpus.get_openpose(openpose_path=filepath, fps=50, people=people_to_extract)

            for person in poses.keys():
                self.assertTrue(person[0] in people_to_extract)


'''
class DgsCorpusTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for dgs_corpus dataset."""

    # TODO(dgs_corpus):
    DATASET_CLASS = dgs_corpus.DgsCorpus
    SPLITS = {
        "train": 3,  # Number of fake train example
        "test": 1,  # Number of fake test example
    }

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
    # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}
'''

if __name__ == "__main__":
    tfds.testing.test_main()
