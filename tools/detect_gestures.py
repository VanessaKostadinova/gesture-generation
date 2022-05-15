import configparser
import json
import os
import matplotlib.pyplot as plt
from os.path import exists
from scipy.signal import savgol_filter
from bvh import Bvh
import numpy as np


# recursive algorithm for creating abstraction of graph with only peaks and troughs
def create_graph_abstraction(gradients, array, i=1, t_0=None, vel=None):
    # find start (where m is negative)
    # go until m becomes positive
    # index is t_0
    # go until m becomes negative
    # index is peak
    # go until m becomes positive
    # index is t_1 and next t_0

    if t_0 is None:
        vel = "up" if gradients[1] - gradients[0] >= 0 else "down"

        while i < len(gradients):
            # print(i < len(gradients))
            vel = "up" if gradients[i] - gradients[i - 1] >= 0 else "down"
            if vel == "up":
                break
            else:
                i += 1
        t_0 = i

    if i == len(gradients):
        return array

    while i < len(gradients):
        vel = "up" if gradients[i] - gradients[i - 1] >= 0 else "down"
        if vel == "down":
            break
        else:
            i += 1
    p = i

    if i == len(gradients):
        return array

    while i < len(gradients):
        vel = "up" if gradients[i] - gradients[i - 1] >= 0 else "down"
        if vel == "up":
            break
        else:
            i += 1
    t_1 = i

    if i == len(gradients):
        return array

    array.append(
        {"x": {"t_0": t_0, "p": p, "t_1": t_1}, "y": {"t_0": gradients[t_0], "p": gradients[p], "t_1": gradients[t_1]}})
    t_0 = t_1
    array = create_graph_abstraction(gradients, array, i, t_0, vel)

    return array


def calculate_every_change(gradients):
    array = []
    vel = "up" if gradients[1] - gradients[0] >= 0 else "down"
    for i in range(2, len(gradients)):
        m = "up" if gradients[i] - gradients[i - 1] >= 0 else "down"
        if not vel == m:
            if m == "down":
                array.append(["peak", i, gradients[i]])
            else:
                array.append(["trough", i, gradients[i]])
            vel = m
    return array


config = configparser.ConfigParser()
config.read("../../config.ini")
paths = config["DEFAULT"]

root_path = paths["OpenPoseDataPath"]
skeleton_path = paths["SkeletonPath"]

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

min_peak_y_delta = 0.4
max_trough_y_delta = 0.3  # of peak height
for d in os.listdir(root_path):
    bvh_path = root_path + d + "\\"
    bvh_name = "hand.bvh"
    if exists(bvh_path + "combined.bvh"):

        with open(bvh_path + "combined.bvh", "r") as bvh_file:
            bvh_lines = bvh_file.readlines()

        diff_per_frame = []

        for c in range(1023, len(bvh_lines)):
            previous_line = bvh_lines[c - 1]
            previous_contents = previous_line.split(" ")[:-1]
            current_line = bvh_lines[c]
            current_contents = current_line.split(" ")[:-1]

            frame_diff = []

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
        plt.plot(smooth_gradients)

        peaks = create_graph_abstraction(smooth_gradients, [])
        filtered = [x for x in peaks if x["y"]["p"] >= 2]
        simple_array = calculate_every_change(smooth_gradients)
        if simple_array[0][0] == "trough":
            start_i = 0
        else:
            start_i = 1

        gesture_range = range(20, 50)
        possible_gestures = []

        # for every possible point
        for i in range(start_i, len(simple_array), 2):
            i_0 = simple_array[i]
            x_0 = i_0[1]
            y_0 = i_0[2]

            # for subsequent point
            for k in range(i, len(simple_array), 2):
                i_1 = simple_array[k]
                x_1 = i_1[1]
                y_1 = i_1[2]

                # if gesture not within min and max range, continue
                if not x_1 - x_0 in gesture_range:
                    continue

                list_range = [x for x in simple_array if (x[0] == "peak") & (x[1] in range(x_0, x_1))]
                tallest_peak = max(list_range, key=lambda x: x[2])

                # compute whether valid
                if abs(y_1 - y_0) >= (float(tallest_peak[2]) * max_trough_y_delta):
                    if (tallest_peak[2] - ((y_1 + y_0) / 2)) >= min_peak_y_delta:
                        possible_gestures.append({"t_0": i_0, "p": tallest_peak, "t_1": i_1})

        # dump as json
        if exists(bvh_path + "gestures.json"):
            open_type = "w"
        else:
            open_type = "x"

        with open(bvh_path + "gestures.json", open_type) as out_file:
            json.dump(possible_gestures, out_file)
