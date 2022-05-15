import os
import configparser
from os.path import exists

config = configparser.ConfigParser()
config.read("../../config.ini")
paths = config["DEFAULT"]

root_path = paths["OpenPoseDataPath"]

c = 0

for d in os.listdir(root_path):
    frame_path = root_path + d + "\\"

    for f in os.listdir(frame_path):
        if f.endswith("_keypoints.json"):
            c += 1

print(c)