import csv
import io
import zipfile

import numpy as np
from pose_format.numpy import NumPyPoseBody
from tqdm import tqdm

import webvtt

def convert_time(vtt_time):
    hhmmss, fraction = vtt_time.split('.')
    h, m, s = hhmmss.split(':')
    hhmmss = int(h) * 3600 + int(m) * 60 + int(s)
    return hhmmss + float(int(fraction) / 1000)

def read_pose_tsv_file(pose_tsv: bytes, num_keypoints: int = 33):
    # TSV file where the first cell is the frame ID
    # and the rest of the cells are the pose data (x, y, z, confidence) for every landmark
    rows = pose_tsv.decode('utf-8').strip().split('\n')
    rows = [row.strip().split('\t') for row in rows]

    num_frames = max(int(row[0]) for row in rows)
    tensor = np.zeros((num_frames + 1, num_keypoints, 4), dtype=np.float32)

    for row in rows:
        frame_id = int(row[0])
        pose_data = row[1:]
        if len(pose_data) > 0:
            vec = np.array(pose_data).reshape(-1, 4)
            if len(vec) != num_keypoints:
                print(f"Warning: pose data has wrong number of keypoints ({len(vec)} instead of {num_keypoints})")
            if num_keypoints == 21:  # hands
                vec[:, 3] = 1
            tensor[frame_id] = vec

    return tensor


def load_pose_body_from_zip(zip_file, fps: float) -> NumPyPoseBody:
    face, left_hand, n, body, right_hand = sorted(zip_file.namelist())
    left_hand_tensor = read_pose_tsv_file(zip_file.read(left_hand), 21)
    right_hand_tensor = read_pose_tsv_file(zip_file.read(right_hand), 21)
    body_tensor = read_pose_tsv_file(zip_file.read(body), 33)
    # TODO: we can add the face if needed
    pose_tensor = np.expand_dims(np.concatenate([body_tensor, left_hand_tensor, right_hand_tensor], axis=1), axis=1)

    data = pose_tensor[:, :, :, :3]
    confidence = np.round(pose_tensor[:, :, :, 3])
    return NumPyPoseBody(fps=fps,
                         data=data,
                         confidence=confidence)


def read_mediapi_set(mediapi_path: str, pose_path: str = None, split='test', pose_type="holistic"):
    pose_zip = "mediapipe_zips.zip" if pose_type == "holistic" else "openpose_zips.zip"
    mediapipe_zips = f'mediapi-skel/7/data/{pose_zip}'
    subtitle_zips = 'mediapi-skel/7/data/subtitles.zip'
    information = 'mediapi-skel/7/information/video_information.csv'

    with zipfile.ZipFile(mediapi_path, 'r') as root_zip:
        # Open the information csv using DictReader
        information_info = root_zip.getinfo(information)
        with root_zip.open(information_info, 'r') as information_file:
            text = io.TextIOWrapper(information_file)
            reader = csv.DictReader(text)
            split_data = [row for row in reader if row['train/dev/test'] == split]

            print("Sample test data:", split_data[0])

        name_number = lambda name: int(name.split('/')[-1].split('.')[0]) if not name.endswith('/') else -1

        # Open the subtitles zip and extract the subtitles for every test datum
        subtitle_info = root_zip.getinfo(subtitle_zips)
        with root_zip.open(subtitle_info, 'r') as subtitle_file:
            nested_zip = zipfile.ZipFile(subtitle_file, 'r')
            number_name = {name_number(name): name for name in nested_zip.namelist()}

            for datum in split_data:
                webvtt_text = nested_zip.read(number_name[int(datum['video'])])
                buffer = io.StringIO(webvtt_text.decode('utf-8'))
                datum['subtitles'] = list(webvtt.read_buffer(buffer))
                datum['subtitles'] = [{"start_time": convert_time(c.start), "end_time": convert_time(c.end), "text": c.text} for c in datum["subtitles"]]


            print("Sample subtitle:", split_data[0]['subtitles'])

        # Open the mediapipe zips and extract the mediapipe data for every test datum
        if pose_path is None:
            print("No pose path given. This means the extraction of mediapipe data will be slow. "
                  "Consider extracting the poses zip using the following command:"
                  "unzip -j mediapi-skel.zip mediapi-skel/7/data/mediapipe_zips.zip -d ."
                  "And then pass MEDIAPI_POSE_PATH=mediapipe_zips.zip to this script.")
            mediapipe_info = root_zip.getinfo(mediapipe_zips)
            mediapipe_file = root_zip.open(mediapipe_info, 'r')
            mediapipe_zip = zipfile.ZipFile(mediapipe_file, 'r')
        else:
            mediapipe_zip = zipfile.ZipFile(pose_path, 'r')

        print("mediapipe files", mediapipe_zip.namelist())
        number_name = {name_number(name): name for name in mediapipe_zip.namelist() if '/00001/' not in name}
        for datum in split_data:
            with zipfile.ZipFile(mediapipe_zip.open(number_name[int(datum['video'])]), 'r') as nested_zip:
                datum['id'] = str(datum['video'])
                datum['metadata'] = {
                    'fps': float(datum['fps'].replace(',', '.')),
                    'duration': float(datum['fps'].replace(',', '.')),
                    'frames': int(datum['frames']),
                    'height': int(datum['height']),
                    'width': int(datum['width']),
                }
                if pose_type is not None:
                    try:
                        datum['pose'] = load_pose_body_from_zip(nested_zip, fps=datum['metadata']['fps'])
                    except Exception as e:
                        print("Failed to load pose for", datum['id'], e)
                        print(datum)
                        continue
                yield datum
