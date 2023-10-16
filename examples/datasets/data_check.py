# from pose_format import Pose
# from sign_language_datasets.datasets.dgs_corpus.dgs_corpus import _POSE_HEADERS
# from sign_language_datasets.utils.features import PoseFeature
#
# files = [
#     "/home/nlp/amit/tensorflow_datasets/downloads/nlp.biu.ac.il_amit_data_pose_holi_dgs_coreCZFUT1DjfgqeTJH8Be7LXAdZjfIcl6MFu9-a0QsMiM.pose",
#     f"/home/nlp/amit/WWW/datasets/poses/holistic/dgs_corpus/1248625-15324720-15465943_a.pose",
#     "/home/nlp/amit/tensorflow_datasets/downloads/nlp.biu.ac.il_amit_data_pose_holi_dgs_corQoeprd_NyhZVoKS6jLUg0R4Yjart_u7A0LK2dpvd2f4.pose",
#     f"/home/nlp/amit/WWW/datasets/poses/holistic/dgs_corpus/1248625-15324720-15465943_b.pose",
# ]
#
# for f_name in files:
#     with open(f_name, "rb") as f:
#         pose = Pose.read(f.read())
#
#     print("fps", pose.body.fps)
#     print("length", len(pose.body.data))
#
#     stride = 50 / 25
#     pose_feature = PoseFeature(shape=(None, 1, 543, 3), stride=stride, header_path=_POSE_HEADERS["holistic"])
#
#     encoded = pose_feature.encode_example(pose)
#     print("fps", encoded["fps"])
#     print("shape", encoded["data"].shape)
#     print("")
#     # [00:32<04:31,  1.90it/s]{'id': '1247205_b', 'fps': 25, 'shape': (868, 1, 75, 3), 'length': 868, 'timestamps': tensor(34.6800)}
#
#     # fps 50
#     # length 42623
#     # fps 25
#     # shape (21312, 1, 543, 3)
#     #
#     # fps 50
#     # length 42623
#     # fps 25
#     # shape (21312, 1, 543, 3)
import json
import os

directory = "/home/nlp/amit/WWW/datasets/poses/holistic/dgs_corpus/"

# Get list of files in the directory
files = os.listdir(directory)

# Create empty dictionary
file_sizes = {}

# Loop through each file in the directory
for file in files:
    # Get full file path
    file_path = os.path.join(directory, file)

    # Get size of file in bytes
    file_size = os.path.getsize(file_path)

    # Add file name and size to dictionary
    file_sizes["https://nlp.biu.ac.il/~amit/datasets/poses/holistic/dgs_corpus/" + file] = file_size

# Print the dictionary
print(file_sizes)

json.dump(file_sizes, open("holistic_pose_size.json", "w"))