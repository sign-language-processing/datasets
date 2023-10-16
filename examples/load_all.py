# import itertools
#
# import tensorflow_datasets as tfds
# from dotenv import load_dotenv
#
# # noinspection PyUnresolvedReferences
# # from sign_language_datasets.datasets.dgs_corpus import DgsCorpusConfig
# from sign_language_datasets.datasets.dgs_corpus import DgsCorpusConfig
#
# import sign_language_datasets.datasets
#
# # noinspection PyUnresolvedReferences
# # import sign_language_datasets.datasets.dgs_corpus
# from sign_language_datasets.datasets.config import SignDatasetConfig
# # from sign_language_datasets.datasets.dgs_corpus.dgs_corpus import DgsCorpusConfig
#
# load_dotenv()
#
# # config = SignDatasetConfig(name="only-annotations", version="3.0.0", include_video=False)
# # rwth_phoenix2014_t = tfds.load(name="rwth_phoenix2014_t", builder_kwargs=dict(config=config))
#
# # config = SignDatasetConfig(name="256x256:10", include_video=True, fps=10, resolution=(256, 256))
#
# # aslg_pc12 = tfds.load('aslg_pc12')
# #
# # rwth_phoenix2014_t = tfds.load('rwth_phoenix2014_t', builder_kwargs=dict(config=config))
#
# # wlasl = tfds.load('wlasl', builder_kwargs=dict(config=config))
# #
# # autsl = tfds.load('autsl', builder_kwargs=dict(
# #     config=SignDatasetConfig(name="test", include_video=False, include_pose="holistic"),
# # ))
#
# # dgs_config = DgsCorpusConfig(name="sentence-test-video", data_type="sentence",
# #                              include_video=False, process_video=False, include_pose=None)
# # dgs_corpus = tfds.load('dgs_corpus', builder_kwargs=dict(config=dgs_config))
# #
# # for datum in itertools.islice(dgs_corpus["train"], 0, 10):
# #   print(datum)
#
#
# config = SignDatasetConfig(name="signbank-annotations", version="1.0.0", include_video=False)
# signbank = tfds.load('sign_bank', builder_kwargs=dict(config=config))
#
# # config = SignDatasetConfig(name="signsuisse3", version="1.0.0", include_video=False, include_pose="holistic")
# # signsuisse = tfds.load('sign_suisse', builder_kwargs=dict(config=config))
#
# # print([d["p.ose"]["data"].shape for d in iter(autsl["train"])])
# # print([d["video"].shape for d in iter(autsl["train"])])
#
# # config = SignDatasetConfig(name="include4", version="1.0.0", extra={"PHPSESSID": "hj9co07ct7f5noq529no9u09l4"})
# # signtyp = tfds.load(name='sign_typ', builder_kwargs=dict(config=config))
# #
# # for datum in itertools.islice(signtyp["train"], 0, 10):
# #   print(datum['sign_writing'].numpy().decode('utf-8'), datum['video'].numpy().decode('utf-8'))
# #
# # config = SignDatasetConfig(name="poses_1", version="1.0.0", include_video=False, include_pose="holistic")
# # dicta_sign = tfds.load(name='dicta_sign', builder_kwargs={"config": config})
#
# # config = SignDatasetConfig(name="only-annotations5", version="1.0.0", include_video=False, process_video=False, include_pose="holistic")
# # dataset = tfds.load(name='sign_bank', builder_kwargs=dict(config=SignDatasetConfig(name="annotations")))
# #
# # decode_str = lambda s: s.numpy().decode('utf-8')
# # for datum in itertools.islice(dataset["train"], 0, 10):
# #     hamnosys = decode_str(datum['hamnosys'])
# #     glosses = [decode_str(g) for g in datum["glosses"]]
# #     print(hamnosys, glosses)
#
#
# #
# # import tensorflow_datasets as tfds
# # # noinspection PyUnresolvedReferences
# # import sign_language_datasets.datasets
# # from sign_language_datasets.datasets.config import SignDatasetConfig
# #
# # # Populate your access tokens
# # TOKENS = {
# #     "zenodo_focusnews_token": "TODO",
# #     "zenodo_srf_videos_token": "TODO",
# #     "zenodo_srf_poses_token": "TODO"
# # }
# #
# # # Load only the annotations, and include path to video files
# # config = SignDatasetConfig(name="annotations", version="1.0.0", process_video=False)
# # wmtslt = tfds.load(name='wmtslt', builder_kwargs={"config": config, **TOKENS})
# #
# # # Load the annotations and openpose poses
# # config = SignDatasetConfig(name="openpose", version="1.0.0", process_video=False, include_pose='openpose')
# # wmtslt = tfds.load(name='wmtslt', builder_kwargs={"config": config, **TOKENS})
# #
# # # Load the annotations and mediapipe holistic poses
# # config = SignDatasetConfig(name="holistic", version="1.0.0", process_video=False, include_pose='holistic')
# # wmtslt = tfds.load(name='wmtslt', builder_kwargs={"config": config, **TOKENS})
# #
# # # Load the full video frames as a tensor
# # config = SignDatasetConfig(name="videos", version="1.0.0", process_video=True)
# # wmtslt = tfds.load(name='wmtslt', builder_kwargs={"config": config, **TOKENS})
# #
# decode_str = lambda s: s.numpy().decode('utf-8')
# for datum in itertools.islice(signsuisse["train"], 0, 10):
#     print(datum)
#     print(datum["pose"])
#     print('\n')
#
#
#

import tensorflow_datasets as tfds
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig

import itertools
#
# config = SignDatasetConfig(name="holistic-poses", version="3.0.0", include_video=False, include_pose="holistic")
# rwth_phoenix2014_t = tfds.load(name='rwth_phoenix2014_t', builder_kwargs=dict(config=config))
#
# for datum in itertools.islice(rwth_phoenix2014_t["train"], 0, 10):
#     print(datum['gloss'].numpy().decode('utf-8'))
#     print(datum['text'].numpy().decode('utf-8'))
#     print(datum['pose']['data'].shape)
#     print()


config = SignDatasetConfig(name="holistic-poses", version="1.0.0", include_video=False, include_pose="holistic")
mediapi_skel = tfds.load(name='mediapi_skel', builder_kwargs=dict(config=config))

for datum in itertools.islice(mediapi_skel["test"], 0, 10):
    print(datum['id'].numpy().decode('utf-8'))
    print(datum['subtitles'])
    print(datum['pose']['data'].shape)
    print()
