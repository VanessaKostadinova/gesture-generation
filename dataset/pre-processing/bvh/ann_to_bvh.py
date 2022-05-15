import configparser
import os
from os.path import exists

from bvh import Bvh

config = configparser.ConfigParser()
config.read("../../config.ini")
paths = config["DEFAULT"]

root_path = paths["OpenPoseDataPath"]
skeleton_path = paths["SkeletonPath"]

ann_name = "ML_out"
ann_extension = ".csv"
out_path = paths["MLOut"]

if exists(out_path + ann_name + ann_extension):
    new_name = out_path + "ML_out.bvh"

    print(new_name)

    with open(out_path + ann_name + ann_extension, "r") as ann_file:
        ann_data = [x.strip() for x in ann_file.readlines()]

    with open("F:\\fyp\\bvh_skeleton", "r") as skeleton:
        skeleton_bvh = Bvh(skeleton.read())
        skeleton.seek(0)
        skeleton_lines = skeleton.readlines()

    right_arm_start_index = skeleton_bvh.get_joint_channels_index("rCollar")
    right_arm_end_index = skeleton_bvh.get_joint_channels_index("rHand") + 2
    left_arm_start_index = skeleton_bvh.get_joint_channels_index("lCollar")
    left_arm_end_index = skeleton_bvh.get_joint_channels_index("lHand") + 2

    # uncomment for hand support
    #right_hand_start_index = skeleton_bvh.get_joint_channels_index("metacarpal1.r")
    #right_hand_end_index = skeleton_bvh.get_joint_channels_index("lCollar") - 1
    #left_hand_start_index = skeleton_bvh.get_joint_channels_index("metacarpal1.l")
    #left_hand_end_index = skeleton_bvh.get_joint_channels_index("rButtock") - 1

    end_channel = skeleton_bvh.get_joint_channels_index("toe5-3.L") + 4

    if exists(new_name):
        open_type = "w"
    else:
        open_type = "x"

    with open(new_name, open_type) as out_file:
        # copy skeleton and frame information
        out_file.writelines(skeleton_lines)
        out_file.write("\n")
        out_file.write("MOTION")
        out_file.write("\n")
        out_file.write("Frames: " + str(len(ann_data)))
        out_file.write("\n")
        out_file.write("Frame Time: 0.04")
        out_file.write("\n")

        for c in range(0, len(ann_data)):
            new_contents = ["0.0"] * end_channel
            ann_contents = ann_data[c].split(" ")
            print(ann_contents)
            # clear root transforms
            # for i in range(0, 6):
            #    arm_content[i] = "0"

            ann_contents_pointer = 0

            # copy right side
            for i in range(right_arm_start_index, right_arm_end_index):
                new_contents[i] = ann_contents[ann_contents_pointer]
                ann_contents_pointer += 1

            # uncomment for hand gesture support
            #for i in range(right_hand_start_index, right_hand_end_index):
            #    new_contents[i] = ann_contents[ann_contents_pointer]
            #    ann_contents_pointer += 1

            # copy left side
            for i in range(left_arm_start_index, left_arm_end_index):
                new_contents[i] = ann_contents[ann_contents_pointer]
                ann_contents_pointer += 1

            # uncomment for hand gesture support
            #for i in range(left_hand_start_index, left_hand_end_index):
            #    new_contents[i] = ann_contents[ann_contents_pointer]
            #    ann_contents_pointer += 1

            for i in range(0, len(new_contents)):
                out_file.write(new_contents[i])
                if c != len(new_contents) - 1:
                    out_file.write(" ")

            if c != len(ann_data) - 1:
                out_file.write("\n")

