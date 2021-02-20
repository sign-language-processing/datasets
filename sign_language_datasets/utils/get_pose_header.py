import os

from pose_format import Pose

p = "/home/nlp/amit/sign-language/sign-language-datasets/old/autsl/holistic/train/"
f_path = p + os.listdir(p)[0]

f = open(f_path, "rb").read()
pose = Pose.read(f)

with open("../datasets/autsl/pose.poseheader", "wb") as fw:
    pose.header.write(fw)
