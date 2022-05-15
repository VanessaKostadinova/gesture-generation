import configparser
import os
from os.path import exists
from bvh import Bvh

config = configparser.ConfigParser()
config.read("../../config.ini")
paths = config["DEFAULT"]

root_path = paths["OpenPoseDataPath"]
skeleton_path = paths["SkeletonPath"]

bvh_arm_name = "arm"
bvh_hand_name = "hand_processed"
bvh_extension = ".bvh"

for d in os.listdir(root_path):
    bvh_path = root_path + d + "\\"
    if exists(bvh_path + bvh_arm_name + bvh_extension) & exists(bvh_path + bvh_hand_name + bvh_extension):
        with open(bvh_path + bvh_arm_name + bvh_extension, "r") as arm_file:
            arm_lines = arm_file.readlines()
        with open(bvh_path + bvh_hand_name + bvh_extension, "r") as hand_file:
            hand_lines = hand_file.readlines()

        new_name = bvh_path + "combined.bvh"

        if exists(new_name):
            open_type = "w"
        else:
            open_type = "x"

        out_file = open(new_name, open_type)

        with open(skeleton_path, "r") as skeleton:
            skeleton_bvh = Bvh(skeleton.read())

        right_arm_start_index = skeleton_bvh.get_joint_channels_index("rCollar")
        right_arm_end_index = skeleton_bvh.get_joint_channels_index("rHand") + 2
        left_arm_start_index = skeleton_bvh.get_joint_channels_index("lCollar")
        left_arm_end_index = skeleton_bvh.get_joint_channels_index("lHand") + 2

        right_hand_start_index = skeleton_bvh.get_joint_channels_index("metacarpal1.r")
        right_hand_end_index = skeleton_bvh.get_joint_channels_index("lCollar") - 1
        left_hand_start_index = skeleton_bvh.get_joint_channels_index("metacarpal1.l")
        left_hand_end_index = skeleton_bvh.get_joint_channels_index("rButtock") - 1

        # copy skeleton and frame information
        for c in range(0, 1021):
            out_file.write(arm_lines[c])

        for c in range(1022, len(arm_lines)):
            arm_content = arm_lines[c].split(" ")
            hand_content = hand_lines[c - 1022].split(" ")
            # clear root transforms
            for i in range(0, 6):
                arm_content[i] = "0"

            # copy right side
            for i in range(right_hand_start_index, right_hand_end_index):
                arm_content[i] = hand_content[i]

            # copy left side
            for i in range(left_hand_start_index, left_hand_end_index):
                arm_content[i] = hand_content[i]

            for i in range(0, len(arm_content)):
                out_file.write(arm_content[i])

                if not i == len(arm_content) - 1:
                    out_file.write(" ")
