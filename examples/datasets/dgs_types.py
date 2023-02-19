import json
import re

import tensorflow_datasets as tfds
import hashlib

# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig


def clear_gloss(gloss: str):
    text = re.search(r'^(.*?)(?=\d|$)', gloss)
    word = text.group(1)
    return word[0].upper() + word[1:].lower()


def dgs_types():
    config = SignDatasetConfig(name="firestore", version="1.0.0", include_video=False)
    dataset = tfds.load(name='dgs_types', builder_kwargs={"config": config})

    for datum in dataset["train"]:
        uid_raw = "dgs_types_" + datum['id'].numpy().decode('utf-8')

        glosses = [clear_gloss(g.numpy().decode('utf-8')) for g in datum['glosses']]
        captions = [{"language": "de", "transcription": g} for g in glosses]
        captions.append({"language": "hns", "transcription": datum['hamnosys'].numpy().decode('utf-8')})

        views = {view.numpy().decode('utf-8'): {
            "video": video.numpy().decode('utf-8'),
            "pose": None
        } for view, video in zip(datum["views"]["name"], datum["views"]["video"])}

        id_func = lambda view_name: 'dgstypes' + hashlib.md5((uid_raw + view_name).encode()).hexdigest()
        ids = [id_func(view_name) for view_name in views.keys()]

        for view_name, view_data in views.items():
            this_id = id_func(view_name)
            doc = {
                "uid": this_id,
                "url": view_data["video"],
                "meta": {
                    "visibility": "unlisted",
                    "name": f"DGS Types: {' / '.join(glosses)} ({view_name})",
                    "language": 'gsg',  # German Sign Language
                    "userId": "dxnkijRxuTP5JEQMaJu88Cxxlu52"  # DGS Account
                },
                "related": [_id for _id in ids if _id != this_id],
                "context": {
                    "id": uid_raw
                }
            }

            yield doc, captions, None


if __name__ == '__main__':
    with open('dgs_types.jsonl', 'w') as f:
        for doc, captions, _ in dgs_types():
            f.write(json.dumps({"doc": doc, "captions": captions}))
            f.write('\n')
