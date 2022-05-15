import os
import json
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

csv_path = "F:\\fyp\\filtered_data\\"

for f in os.listdir(csv_path):
    if f.endswith(".csv"):
        csv_file = open(csv_path + f)
        file_name = f[:-4]

        if os.path.exists("F:\\fyp\\raw_data\\" + file_name + ".mp4") & os.path.exists(csv_path + file_name + "_aux_info.json"):
            json_file = json.load(open(csv_path + file_name + "_aux_info.json"))
            next(csv_file)
            next(csv_file)
            line_counter = 0
            for line_number in range(0, len(json_file) - 1):
                line = next(csv_file)
                if json_file[line_counter]["message"] == "PASS":
                    values = line.split(",")
                    ffmpeg_extract_subclip("F:\\fyp\\raw_data\\" + file_name + ".mp4",
                                           float(values[3]),
                                           float(values[3])+float(values[4]),
                                           "F:\\fyp\\cut_data_2\\" + values[0] + "_" + file_name + ".mp4")
                line_counter += 1

