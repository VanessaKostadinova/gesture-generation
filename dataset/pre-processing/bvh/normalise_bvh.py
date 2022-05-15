import configparser
import os
from os.path import exists

config = configparser.ConfigParser()
config.read("../../config.ini")
paths = config["DEFAULT"]

root_path = paths["OpenPoseDataPath"]
bvh_name = "arm"
bvh_extension = ".bvh"

for d in os.listdir(root_path):
    bvh_path = root_path + d + "\\"
    if exists(bvh_path + bvh_name + bvh_extension):
        print(bvh_path)
        with open(bvh_path + bvh_name + bvh_extension, "r") as bvh_file:
            bvh_lines = bvh_file.readlines()

        new_name = bvh_path + bvh_name + "_normalised.bvh"

        if exists(new_name):
            open_type = "w"
        else:
            open_type = "x"

        out_file = open(new_name, open_type)

        # copy skeleton and frame information
        for c in range(0, 1021):
            out_file.write(bvh_lines[c])

        for c in range(1022, len(bvh_lines)):
            bvh_content = bvh_lines[c].split(" ")
            # clear root transforms
            for i in range(0, 6):
                bvh_content[i] = "0"

            for i in range(0, len(bvh_content)):
                out_file.write(bvh_content[i])

                if not i == len(bvh_content) - 1:
                    out_file.write(" ")
