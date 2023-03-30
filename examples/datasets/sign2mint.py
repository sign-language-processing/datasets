import json

import tensorflow_datasets as tfds
import hashlib

# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig

def sign2mint():
    config = SignDatasetConfig(name="firestore", version="1.0.0", include_video=False)
    dataset = tfds.load(name='sign2_mint', builder_kwargs={"config": config})

    for datum in dataset["train"]:
        name = datum['fachbegriff'].numpy().decode('utf-8')
        swu = datum['swu'].numpy().decode('utf-8')
        url = datum['video'].numpy().decode('utf-8')
        uid_raw = "sign2mint" + datum['id'].numpy().decode('utf-8')
        uid = 's2m' + hashlib.md5(uid_raw.encode()).hexdigest()

        doc = {
            "uid": uid,
            "url": url,
            "meta": {
                "visibility": "unlisted",
                "name": "Sign2MINT: " + name,
                "language": "gsg",
                "userId": "1aGEJfRxDqdpIbOnjU5i5X0quYu2"
            },
            "context": {
                "id": uid_raw,
                "wiktionary": datum['wortlink'].numpy().decode('utf-8'),
                "definition": datum['definition'].numpy().decode('utf-8')
            }
        }

        captions = [
            {"language": "de", "transcription": name},
            {"language": "Sgnw", "transcription": swu},
        ]

        yield doc, captions, None


if __name__ == '__main__':
    with open('sign2mint.jsonl', 'w') as f:
        for doc, captions, _ in sign2mint():
            f.write(json.dumps({"doc": doc, "captions": captions}))
            f.write('\n')
