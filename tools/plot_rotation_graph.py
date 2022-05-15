import configparser
import os
import matplotlib.pyplot as plt
from os.path import exists
from scipy.signal import savgol_filter
from bvh import Bvh
import numpy as np

config = configparser.ConfigParser()
config.read("../../config.ini")
paths = config["DEFAULT"]

root_path = paths["OpenPoseDataPath"]

with open("F:\\fyp\\bvh_skeleton", "r") as skeleton:
    skeleton_bvh = Bvh(skeleton.read())

right_arm_start_index = skeleton_bvh.get_joint_channels_index("rCollar")
right_arm_end_index = skeleton_bvh.get_joint_channels_index("rHand") + 2
left_arm_start_index = skeleton_bvh.get_joint_channels_index("lCollar")
left_arm_end_index = skeleton_bvh.get_joint_channels_index("lHand") + 2

right_hand_start_index = skeleton_bvh.get_joint_channels_index("metacarpal1.r")
right_hand_end_index = skeleton_bvh.get_joint_channels_index("lCollar") - 1
left_hand_start_index = skeleton_bvh.get_joint_channels_index("metacarpal1.l")
left_hand_end_index = skeleton_bvh.get_joint_channels_index("rButtock") - 1

for d in os.listdir(root_path):
    bvh_path = root_path + d + "\\"
    bvh_name = "hand.bvh"
    if exists(bvh_path + bvh_name):
        print(bvh_path)
        plot_type = "arm"  # change to hand or both to compute only hand diff or both diff
        with open(bvh_path + bvh_name, "r") as bvh_file:
            bvh_lines = bvh_file.readlines()

        diff_per_frame = []

        for c in range(1023, len(bvh_lines)):
            previous_line = bvh_lines[c - 1]
            previous_contents = previous_line.split(" ")[:-1]
            current_line = bvh_lines[c]
            current_contents = current_line.split(" ")[:-1]

            frame_diff = []
            if plot_type == "arm":
                for i in range(left_arm_start_index, left_arm_end_index):
                    frame_diff.append(abs(float(previous_contents[i]) - float(current_contents[i])))

                for i in range(right_arm_start_index, right_arm_end_index):
                    frame_diff.append(abs(float(previous_contents[i]) - float(current_contents[i])))

            elif plot_type == "hand":
                for i in range(left_hand_start_index, left_hand_end_index):
                    frame_diff.append(abs(float(previous_contents[i]) - float(current_contents[i])))

                for i in range(right_hand_start_index, right_hand_end_index):
                    frame_diff.append(abs(float(previous_contents[i]) - float(current_contents[i])))

            elif plot_type == "both":
                for i in range(right_arm_start_index, right_arm_end_index):
                    frame_diff.append(abs(float(previous_contents[i]) - float(current_contents[i])))

                for i in range(right_hand_start_index, right_hand_end_index):
                    frame_diff.append(abs(float(previous_contents[i]) - float(current_contents[i])))

                for i in range(left_arm_start_index, left_arm_end_index):
                    frame_diff.append(abs(float(previous_contents[i]) - float(current_contents[i])))

                for i in range(left_hand_start_index, left_hand_end_index):
                    frame_diff.append(abs(float(previous_contents[i]) - float(current_contents[i])))

            avg_diff = sum(frame_diff) / len(frame_diff)

            diff_per_frame.append(avg_diff)

        smooth_gradients = savgol_filter(np.array(diff_per_frame), 33, 9)
        print(smooth_gradients)
        plt.plot(smooth_gradients)
        plt.show()
