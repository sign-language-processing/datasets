import json

import tensorflow_datasets as tfds
from dotenv import load_dotenv
import hashlib

# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig

load_dotenv()

IANA_TAGS = {
    "ch-de": "sgg",
    "ch-fr": "ssr",
    "ch-it": "slf",
}


def signsuisse():
    config = SignDatasetConfig(name="firestore-4", version="1.0.0", include_video=False)
    dataset = tfds.load(name='sign_suisse', builder_kwargs={"config": config})

    # signbank = tfds.load(name='sign_bank', builder_kwargs={"config": config})

    for datum in dataset["train"]:
        uid_raw = "signsuisse" + datum['id'].numpy().decode('utf-8')

        signed_language_original = datum['signedLanguage'].numpy().decode('utf-8')
        signed_language = IANA_TAGS[signed_language_original]
        category = datum['category'].numpy().decode('utf-8')
        url = datum['url'].numpy().decode('utf-8')
        paraphrase = datum['paraphrase'].numpy().decode('utf-8')
        definition = datum['definition'].numpy().decode('utf-8')

        example_text = datum['exampleText'].numpy().decode('utf-8')
        videos = [{
            "id_suffix": "isolated",
            "text": datum['name'].numpy().decode('utf-8'),
            "url": datum['video'].numpy().decode('utf-8'),
            "example": example_text if len(example_text) > 0 else None
        }]

        example_video_url = datum['exampleVideo'].numpy().decode('utf-8')
        if len(example_text) > 0:
            videos.append({
                "id_suffix": "example",
                "text": example_text,
                "url": example_video_url,
                "example": None
            })

        spoken_language = datum['spokenLanguage'].numpy().decode('utf-8')

        id_func = lambda suffix: 'ss' + hashlib.md5((uid_raw+suffix).encode()).hexdigest()
        ids = [id_func(video["id_suffix"]) for video in videos]

        for video in videos:
            if video["text"].lower() == "mit" or video["example"] == "Kaffee mit Zucker und Milch.":
                print("FOUND", video)

            this_id = id_func(video["id_suffix"])
            doc = {
                "uid": this_id,
                "url": video["url"],
                "meta": {
                    "visibility": "unlisted",
                    "name": "SignSuisse: " + video["text"],
                    "language": signed_language,
                    "userId": "ndHslReGkGSJMHxAoSkBeLfjbfU2"
                },
                "related": [_id for _id in ids if _id != this_id],
                "context": {
                    "id": uid_raw,
                    "category": category,
                    "url": url,
                    "example": video["example"],
                    "paraphrase": paraphrase,
                    "definition": definition
                }
            }

            captions = [
                {"language": spoken_language, "transcription": video["text"]},
            ]

            yield doc, captions, None


if __name__ == '__main__':
    with open('signsuisse.jsonl', 'w') as f:
        for doc, captions, _ in signsuisse():
            f.write(json.dumps({"doc": doc, "captions": captions}))
            f.write('\n')
