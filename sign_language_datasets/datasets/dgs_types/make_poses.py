from os import path

import os
import tensorflow_datasets as tfds
from pose_format import Pose
from tqdm import tqdm
from dotenv import load_dotenv

# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets

# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets.dgs_corpus
from sign_language_datasets.datasets.config import SignDatasetConfig

from pose_format.utils.holistic import load_holistic

import cv2


def load_video(cap: cv2.VideoCapture):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


config = SignDatasetConfig(name="full3", version="1.0.0", include_video=True, process_video=False, include_pose=None)
dgs_types = tfds.load(name='dgs_types', builder_kwargs={"config": config})

decode_str = lambda s: s.numpy().decode('utf-8')
for datum in tqdm(dgs_types["train"]):
    _id = decode_str(datum['id'])
    for view, video in zip(datum["views"]["name"], datum["views"]["video"]):
        view = decode_str(view)
        video = decode_str(video)

        pose_file_path = path.join("/home/nlp/amit/WWW/datasets/poses/holistic/dgs_types", f"{_id}_{view}.pose")
        if path.exists(pose_file_path):
            with open(pose_file_path, "rb") as f:
                pose = Pose.read(f.read())
            if pose.body.data.shape[2] != 576:
                print(f"Skipping {_id}_{view} because it has {pose.body.shape[2]} landmarks instead of 576")
                os.remove(pose_file_path)

        if not path.exists(pose_file_path):
            print(f"Processing {_id}_{view}...")
            cap = cv2.VideoCapture(video)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            all_frames = list(load_video(cap))

            pose = load_holistic(all_frames, fps=fps, width=width, height=height, depth=width, progress=False)
            with open(pose_file_path, "wb") as f:
                pose.write(f)

            cap.release()



with open("holistic.poseheader", "wb") as f:
    pose.header.write(f)