import os

open_pose_dir = "C:\\Users\\vanes\\Downloads\\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\\openpose"
params = "--number_people_max 1 --hand --net_resolution -256x128"

video_dir = "F:\\fyp\\cut_data_2\\"
video_name = ""

data_output_dir = "F:\\fyp\\cut_data_openpose\\"
data_output_file = ""

if not os.path.exists(data_output_dir):
    os.mkdir(data_output_dir)

for v in os.listdir(video_dir):
    if v.endswith(".mp4"):
        data_output_file = v[:-4]
        if not os.path.exists(data_output_dir + data_output_file):
            os.mkdir(data_output_dir + data_output_file)
            video_name = v
            command = "cd " + open_pose_dir + " & bin\\OpenPoseDemo.exe --video " + video_dir + video_name + " --write_json " + data_output_dir + data_output_file + " " + params
            print(command)
            os.system('cmd /c "' + command + '"')