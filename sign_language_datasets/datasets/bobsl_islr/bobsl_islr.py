"""BOBSL is a large-scale dataset of British Sign Language (BSL)."""

import csv
import json
import pickle
import os
from os import path
from tqdm import tqdm
from typing import Union
from collections import defaultdict

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from pose_format import Pose

from sign_language_datasets.utils.features import PoseFeature

from ..warning import dataset_warning
from ...datasets.config import SignDatasetConfig, cloud_bucket_file

_DESCRIPTION = """
~5M noisy isolated signs collected from sign spottings on BOBSL
"""

_CITATION = """
@inproceedings{momeni2022automatic,
  title={Automatic dense annotation of large-vocabulary sign language videos},
  author={Momeni, Liliane and Bull, Hannah and Prajwal, KR and Albanie, Samuel and Varol, G{\"u}l and Zisserman, Andrew},
  booktitle={European Conference on Computer Vision},
  pages={671--690},
  year={2022}
}

@Article{Albanie2021bobsl,
    author       = "Samuel Albanie and G{\"u}l Varol and Liliane Momeni and Hannah Bull and Triantafyllos Afouras and Himel Chowdhury and Neil Fox and Bencie Woll and Rob Cooper and Andrew McParland and Andrew Zisserman",
    title        = "{BOBSL}: {BBC}-{O}xford {B}ritish {S}ign {L}anguage {D}ataset",
    howpublished = "https://www.robots.ox.ac.uk/~vgg/data/bobsl",
    year         = "2021",
    journal      = "arXiv"
}
"""

_DOWNLOAD_URL = 'https://www.robots.ox.ac.uk/~vgg/data/bobsl/'

_POSE_HEADERS = {"holistic": path.join(path.dirname(path.realpath(__file__)), "holistic.poseheader")}


class BobslIslr(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Popsign dataset."""

    VERSION = tfds.core.Version("1.4.0")
    RELEASE_NOTES = {"1.4.0": "v1.4"}

    BUILDER_CONFIGS = [SignDatasetConfig(name="default", include_pose="holistic")]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {
            "id": tfds.features.Text(),
            "text": tfds.features.Text(),
            "episode_id": tfds.features.Text(),
            "start_frame": tfds.features.Scalar(dtype=np.int64),
            "end_frame": tfds.features.Scalar(dtype=np.int64),
        }

        # TODO: add videos

        if self._builder_config.include_pose == "holistic":
            pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
            stride = 1 if self._builder_config.fps is None else 25 / self._builder_config.fps
            features["pose"] = PoseFeature(shape=(None, 1, 576, 3), header_path=pose_header_path, stride=stride)

        if "lip_feature_dir" in self._builder_config.extra:
            features["lip"] = tfds.features.Tensor(shape=(None, 768), dtype=np.float32)

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage=_DOWNLOAD_URL,
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dataset_warning(self)

        # too expensive to host the poses at the moment, need to specify a local path
        # poses_dir = str(dl_manager.download_and_extract(_POSE_URLS["holistic"]))
        poses_dir = self._builder_config.extra["poses_dir"]

        print(f'Generating train and valid set ...')
        # Copy aggregated annotations from the vgg_islr repo
        # Originated from K R Prajwal
        # bobsl_unfiltered_mouthings_praj_hannah_sent_plus_dict_ha_align_syn_plus_attention_plus_i3d_top1_pseudo_labels_clean_8697.pkl

        type2offsets = {
            "prajwal_mouthing" : (-9, 11), "dict" : (-3, 22), "attention" : (-8, 18), 
            "i3d_pseudo_label" : (0, 19), "mouthing" : (-15, 4), 
            "swin_pseudo_label" : (5, 25), "other" : (-8, 8), "cos_sim" : (0, 19)
        }
        type2prob_thresh = {
            0 : # train
                {"prajwal_mouthing" : 0.8, "dict" : 0.8, "attention" : 0., "i3d_pseudo_label" : 0.5,
                "swin_pseudo_label" : 0.3, "cos_sim" : 2., "other" : 0.},

            1 : {"prajwal_mouthing" : 0.8, "dict" : 0.9, "attention" : 0., "i3d_pseudo_label" : 0.5,
            "swin_pseudo_label" : 0.3, "cos_sim" : 2., "other" : 0.},

            3 : {"prajwal_mouthing" : 0.8, "dict" : 0.8, "mouthing" : 0.8, "other" : 0.},
        }
        TRAIN_SPLIT_NUM = 0
        VAL_SPLIT_NUM = 1
        vocab_file = "/work/sign-language/haran/bobsl/vocab/8697_vocab.pkl"
        spotting_file = "/work/sign-language/youngjoon/islr/anno.pkl"
        fps = 25

        print('Load vocab ...')
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)["words_to_id"]
            id2word = {id : word for word, id in vocab.items()}
            print('Vocab size:', len(vocab))
        print('Load spotting annotations ...')
        with open(spotting_file, 'rb') as f:
            data = pickle.load(f)
        data = data["videos"]

        examples = {
            TRAIN_SPLIT_NUM: defaultdict(list),
            VAL_SPLIT_NUM: defaultdict(list),
        }
        # for split_idx in [VAL_SPLIT_NUM]:
        for split_idx in [TRAIN_SPLIT_NUM,VAL_SPLIT_NUM]:
            count = 0 

            print('Load examples from anno.pkl ...')
            for i in tqdm(range(len(data["name"]))):
                if data["split"][i] == split_idx:
                    if data["word"][i] in ['_fingerspelling', '_nosigning', '_pointing', '_lexical_signing']:
                        continue

                    if data["word"][i] in vocab:
                        if data["mouthing_prob"][i] >= type2prob_thresh[split_idx][data["anno_type"][i]]:
                            pose_filename = data["name"][i].replace('.mp4', '.pose')
                            pose_path = path.join(poses_dir, pose_filename)

                            if os.path.exists(pose_path):
                                time = int(data["mouthing_time"][i] * fps)
                                start_offset, end_offset = type2offsets[data["anno_type"][i]]
                                s, e = max(0, time + start_offset), time + end_offset

                                w_l = data["word"][i] 
                                w_l = w_l if isinstance(w_l, tuple) else [w_l]

                                for w in w_l:
                                    episode_id = pose_filename.replace('.pose', '')
                                    idx = f"{'train' if split_idx == TRAIN_SPLIT_NUM else 'val'}-{w}-{episode_id}-{i}"
                                    examples[split_idx][episode_id].append({
                                        'idx': idx,
                                        'text': w,
                                        'start_frame': s,
                                        'end_frame': e,
                                        'episode_id': episode_id,
                                    })

                                # DEBUG
                                # count = count + 1 
                                # if count >= 10:
                                #     break
                            else:
                                print(f'{pose_path} does not exist, skipping ...')

        print(f'Generating test set ...')
        # Read the 25K ISOLATED SIGNS annotation files from BOBSL website

        BOBSL_PATH = '/athenahomes/zifan/BOBSL/v1.4'
        annotations = {
            'test': {
                'dict': {
                    'spottings_path': f"{BOBSL_PATH}/manual_annotations/isolated_signs/verified_dict_spottings.json",
                    'range': [-3, 22],
                },
                'mouthing': {
                    'spottings_path': f"{BOBSL_PATH}/manual_annotations/isolated_signs/verified_mouthing_spottings.json",
                    'range': [-15, 4],
                },
            },
        }
        fps = 25

        test_examples = []
        for annotation_source, annotation in annotations['test'].items():
            print(f'Loading {annotation_source} ...')

            file_path = annotation['spottings_path']
            if file_path.endswith('.json'):
                with open(file_path, "r") as file:
                    data = json.load(file)
                    data = data['test']

                    annotation['total_num'] = sum([len(d['names']) for d in data.values()])
                    annotation['vocab'] = len(data)

                    for gloss, value in tqdm(list(data.items())):
                        for i, name in enumerate(value['names']):
                            global_time = value['global_times'][i]
                            filename = f"{name}-{str(global_time).replace('.', '_')}"
                            idx = f"test-{gloss}-{filename}"
                            pose_path = f"{annotation_source}/{gloss}/{filename}.pose"

                            start_offset, end_offset = annotation['range']
                            s, e = max(0, int(global_time * fps + start_offset)), int(global_time * fps + end_offset)

                            test_examples.append({
                                'idx': idx,
                                # 'pose_path': pose_path,
                                'text': gloss,
                                'start_frame': s,
                                'end_frame': e,
                                'episode_id': filename.split('-')[0],
                            })

        idxs = [item["idx"] for item in test_examples]
        assert len(idxs) == len(set(idxs)), "Duplicate 'idx' values found!"
        
        # test_examples = test_examples[:10] # DEBUG

        # Group examples by episode_id
        grouped_examples = defaultdict(list)
        for example in test_examples:
            grouped_examples[example['episode_id']].append(example)
        test_examples = grouped_examples

        print('Train:', sum([len(d) for d in examples[TRAIN_SPLIT_NUM].values()]))
        print('Valid:', sum([len(d) for d in examples[VAL_SPLIT_NUM].values()]))
        print('Test:', sum([len(d) for d in test_examples.values()]))

        return [
            tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"poses_dir": poses_dir, "examples": examples[TRAIN_SPLIT_NUM]}),
            tfds.core.SplitGenerator(name=tfds.Split.VALIDATION, gen_kwargs={"poses_dir": poses_dir, "examples": examples[VAL_SPLIT_NUM]}),
            tfds.core.SplitGenerator(name=tfds.Split.TEST, gen_kwargs={"poses_dir": poses_dir, "examples": test_examples}),
        ]

    def _generate_examples(self, poses_dir: str, examples: dict):
        """Yields examples."""

        lip_dir = self._builder_config.extra["lip_feature_dir"] if "lip_feature_dir" in self._builder_config.extra else None

        for episode_id, episode_examples in examples.items():
            mediapipe_path = path.join(poses_dir, episode_id + '.pose')

            with open(mediapipe_path, "rb") as f:
                buffer = f.read()

                if lip_dir:
                    feat_path = path.join(lip_dir, episode_id + ".npy")
                    lip_feat = np.load(feat_path) if os.path.exists(feat_path) else None
                
                for example in episode_examples:
                    try:
                        datum = {
                            "id": example["idx"], 
                            "text": example["text"],
                            "episode_id": example["episode_id"],
                            "start_frame": example["start_frame"],
                            "end_frame": example["end_frame"],
                        }

                        pose = Pose.read(buffer, start_frame=example["start_frame"], end_frame=example["end_frame"])
                        datum["pose"] = pose
                        
                        if lip_dir:
                            if lip_feat is not None:
                                datum['lip'] = lip_feat[example['start_frame']:example['end_frame']]
                                assert datum['lip'].shape[0] == pose.body.data.shape[0], \
                                    f"lip reading feature should have the same number of frames as pose: {datum['lip'].shape[0]} vs. {pose.body.data.shape[0]}"
                            else:
                                print(f'WARNING: {feat_path} not found ...')
                                datum['lip'] = np.zeros((pose.body.data.shape[0], 768), dtype=np.float32)

                        yield datum["id"], datum
                    except Exception as e:
                        print(e)
