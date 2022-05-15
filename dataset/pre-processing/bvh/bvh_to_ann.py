import os
from os.path import exists

from bvh import Bvh
import globals

root_path = globals.paths["OpenPoseDataPath"]
skeleton_path = globals.paths["SkeletonPath"]

bvh_arm_name = "arm_normalised"
bvh_hand_name = "hand_processed"
bvh_extension = ".bvh"

for d in os.listdir(root_path):
    bvh_path = root_path + d + "\\"
    if exists(bvh_path + bvh_arm_name + bvh_extension) & exists(bvh_path + bvh_hand_name + bvh_extension):
        with open(bvh_path + bvh_arm_name + bvh_extension, "r") as arm_file:
            arm_lines = arm_file.readlines()

        # uncomment for hand support
        # with open(bvh_path + bvh_hand_name + bvh_extension, "r") as hand_file:
        #    hand_lines = hand_file.readlines()

        new_name = bvh_path + "ML_BVH.txt"

        with open(skeleton_path, "r") as skeleton_file:
            skeleton_bvh = Bvh(skeleton_file.read())

        right_arm_start_index = skeleton_bvh.get_joint_channels_index("rCollar")
        right_arm_end_index = skeleton_bvh.get_joint_channels_index("rHand") + 2
        left_arm_start_index = skeleton_bvh.get_joint_channels_index("lCollar")
        left_arm_end_index = skeleton_bvh.get_joint_channels_index("lHand") + 2

        # uncomment for hand support
        # right_hand_start_index = skeleton_bvh.get_joint_channels_index("metacarpal1.r")
        # right_hand_end_index = skeleton_bvh.get_joint_channels_index("lCollar") - 1
        # left_hand_start_index = skeleton_bvh.get_joint_channels_index("metacarpal1.l")
        # left_hand_end_index = skeleton_bvh.get_joint_channels_index("rButtock") - 1

        if exists(new_name):
            open_type = "w"
        else:
            open_type = "x"

        with open(new_name, open_type) as out_file:
            for c in range(1022, len(arm_lines)):
                new_contents = ""
                arm_content = arm_lines[c].split(" ")
                # uncomment for hand support
                # hand_content = hand_lines[c - 1022].split(" ")

                # copy right side
                for i in range(right_arm_start_index, right_arm_end_index):
                    new_contents += arm_content[i] + ","

                # uncomment for hand support
                # for i in range(right_hand_start_index, right_hand_end_index):
                #    new_contents += hand_content[i] + ","

                # copy left side
                for i in range(left_arm_start_index, left_arm_end_index):
                    new_contents += arm_content[i]
                    if not i == left_arm_end_index - 1:
                        new_contents += ","

                # uncomment for hand support
                # for i in range(left_hand_start_index, left_hand_end_index):
                #    new_contents += hand_content[i]

                out_file.write(new_contents)
                if c != len(arm_lines) - 1:
                    out_file.write("\n")
