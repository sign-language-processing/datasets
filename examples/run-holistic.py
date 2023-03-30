import itertools
from os import path

import tensorflow_datasets as tfds
from datasets import tqdm
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


config = SignDatasetConfig(name="full2", version="1.0.0", include_video=True, process_video=False, include_pose=None)
dgs_types = tfds.load(name='dgs_types', builder_kwargs={"config": config})

decode_str = lambda s: s.numpy().decode('utf-8')
for datum in tqdm(dgs_types["train"]):
    _id = decode_str(datum['id'])
    for view, video in zip(datum["views"]["name"], datum["views"]["video"]):
        view = decode_str(view)
        video = decode_str(video)

        pose_file_path = path.join("/home/nlp/amit/WWW/datasets/poses/holistic/dgs_types", f"{_id}_{view}.pose")
        if not path.exists(pose_file_path):
            cap = cv2.VideoCapture(video)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            all_frames = list(load_video(cap))

            pose = load_holistic(all_frames, fps=fps, width=width, height=height, depth=width, progress=False)
            with open(pose_file_path, "wb") as f:
                pose.write(f)

            cap.release()
