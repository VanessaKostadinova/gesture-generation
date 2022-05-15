import os
import globals
from os.path import exists
from bvh import Bvh

root_path = globals.paths["OpenPoseDataPath"]

with open("F:\\fyp\\bvh_skeleton", "r") as skeleton:
    skeleton_bvh = Bvh(skeleton.read())

range_thumb_m = [-10, 55]
range_thumb_p = [-10, 80]

range_index_m = [-10, 90]
range_index_p = [-10, 100]
range_index_d = [-10, 80]

range_middle_m = [-10, 90]
range_middle_p = [-10, 100]
range_middle_d = [-10, 80]

range_ring_m = [-10, 90]
range_ring_p = [-10, 100]
range_ring_d = [-10, 80]

range_little_m = [-10, 90]
range_little_p = [-10, 100]
range_little_d = [-10, 80]

range_l_thumb_m = [55, -10]
range_l_thumb_p = [80, -10]

range_l_index_m = [10, -90]
range_l_index_p = [10, -100]
range_l_index_d = [10, -80]

range_l_middle_m = [10, -90]
range_l_middle_p = [10, -100]
range_l_middle_d = [10, -80]

range_l_ring_m = [10, -90]
range_l_ring_p = [10, -100]
range_l_ring_d = [10, -80]

range_l_little_m = [10, -90]
range_l_little_p = [10, -100]
range_l_little_d = [10, -80]

finger_joints = {
    "r_thumb_m": {"position": skeleton_bvh.get_joint_channels_index("finger1-2.r"), "limit": range_thumb_m},
    "r_thumb_p": {"position": skeleton_bvh.get_joint_channels_index("finger1-3.r"), "limit": range_thumb_p},
    "r_index_m": {"position": skeleton_bvh.get_joint_channels_index("finger2-1.r"), "limit": range_index_m},
    "r_index_p": {"position": skeleton_bvh.get_joint_channels_index("finger2-2.r"), "limit": range_index_p},
    "r_index_d": {"position": skeleton_bvh.get_joint_channels_index("finger2-3.r"), "limit": range_index_d},
    "r_middle_m": {"position": skeleton_bvh.get_joint_channels_index("finger3-1.r"), "limit": range_middle_m},
    "r_middle_p": {"position": skeleton_bvh.get_joint_channels_index("finger3-2.r"), "limit": range_middle_p},
    "r_middle_d": {"position": skeleton_bvh.get_joint_channels_index("finger3-3.r"), "limit": range_middle_d},
    "r_ring_m": {"position": skeleton_bvh.get_joint_channels_index("finger4-1.r"), "limit": range_ring_m},
    "r_ring_p": {"position": skeleton_bvh.get_joint_channels_index("finger4-2.r"), "limit": range_ring_p},
    "r_ring_d": {"position": skeleton_bvh.get_joint_channels_index("finger4-3.r"), "limit": range_ring_d},
    "r_little_m": {"position": skeleton_bvh.get_joint_channels_index("finger5-1.r"), "limit": range_little_m},
    "r_little_p": {"position": skeleton_bvh.get_joint_channels_index("finger5-2.r"), "limit": range_little_p},
    "r_little_d": {"position": skeleton_bvh.get_joint_channels_index("finger5-3.r"), "limit": range_little_d},
    "l_thumb_m": {"position": skeleton_bvh.get_joint_channels_index("finger1-2.l"), "limit": range_l_thumb_m},
    "l_thumb_p": {"position": skeleton_bvh.get_joint_channels_index("finger1-3.l"), "limit": range_l_thumb_p},
    "l_index_m": {"position": skeleton_bvh.get_joint_channels_index("finger2-1.l"), "limit": range_l_index_m},
    "l_index_p": {"position": skeleton_bvh.get_joint_channels_index("finger2-2.l"), "limit": range_l_index_p},
    "l_index_d": {"position": skeleton_bvh.get_joint_channels_index("finger2-3.l"), "limit": range_l_index_d},
    "l_middle_m": {"position": skeleton_bvh.get_joint_channels_index("finger3-1.l"), "limit": range_l_middle_m},
    "l_middle_p": {"position": skeleton_bvh.get_joint_channels_index("finger3-2.l"), "limit": range_l_middle_p},
    "l_middle_d": {"position": skeleton_bvh.get_joint_channels_index("finger3-3.l"), "limit": range_l_middle_d},
    "l_ring_m": {"position": skeleton_bvh.get_joint_channels_index("finger4-1.l"), "limit": range_l_ring_m},
    "l_ring_p": {"position": skeleton_bvh.get_joint_channels_index("finger4-2.l"), "limit": range_l_ring_p},
    "l_ring_d": {"position": skeleton_bvh.get_joint_channels_index("finger4-3.l"), "limit": range_l_ring_d},
    "l_little_m": {"position": skeleton_bvh.get_joint_channels_index("finger5-1.l"), "limit": range_l_little_m},
    "l_little_p": {"position": skeleton_bvh.get_joint_channels_index("finger5-2.l"), "limit": range_l_little_p},
    "l_little_d": {"position": skeleton_bvh.get_joint_channels_index("finger5-3.l"), "limit": range_l_little_d}}

for d in os.listdir(root_path):
    bvh_path = root_path + d + "\\"

    if os.path.exists(bvh_path + "hand.bvh"):
        bvh_file = open(bvh_path + "hand.bvh", "r")

        with open(bvh_path + "hand.bvh", "r") as file:
            bvh_lines = file.readlines()

        if exists(bvh_path + "hand_processed.bvh"):
            open_type = "w"
        else:
            open_type = "x"

        with open(bvh_path + "hand_processed.bvh", open_type) as out_file:
            # first pass to flag all wrong frames
            for c in range(1022, len(bvh_lines)):
                contents = bvh_lines[c].split(" ")

                for joint_name in finger_joints:
                    # print(type(joint))
                    joint = finger_joints[joint_name]
                    position = int(joint["position"])
                    angle = float(contents[position])
                    # if invalid
                    if joint_name.split("_")[0] == "r":
                        if not joint["limit"][0] <= angle <= joint["limit"][1]:
                            inverse_angle = angle * -1
                            # check if we can reverse it
                            if joint["limit"][0] <= inverse_angle <= joint["limit"][1]:
                                # print("true")
                                # print(joint_name)
                                contents[position] = str(inverse_angle)
                            else:
                                # if we can't reverse, set to limit
                                if angle <= joint["limit"][0]:
                                    contents[position] = joint["limit"][0]
                                else:
                                    contents[position] = joint["limit"][1]
                    else:
                        if not joint["limit"][0] >= angle >= joint["limit"][1]:
                            inverse_angle = angle * -1
                            # check if we can reverse it
                            if joint["limit"][0] >= inverse_angle >= joint["limit"][1]:
                                contents[position] = str(inverse_angle)
                            else:
                                # if we can't reverse, set to limit
                                if angle >= joint["limit"][0]:
                                    contents[position] = joint["limit"][0]
                                else:
                                    contents[position] = joint["limit"][1]
                for i in range(0, len(contents)):
                    out_file.write(str(contents[i]))
                    if i != len(contents) - 1:
                        out_file.write(" ")