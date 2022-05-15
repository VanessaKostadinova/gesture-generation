from pytube import YouTube
import os
from pytube.exceptions import VideoPrivate, VideoUnavailable
import csv
from tools import sanitise_filesystem_name

# id_file = open("F:\\fyp\\video_ids.txt", "r")
# id_list = id_file.readlines()


def download_from_id(id_list):
    for li in id_list:
        try:
            yt = YouTube('http://youtube.com/watch?v=' + li)
            yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(
                "F:\\fyp\\raw_data", sanitise_filesystem_name(li) + ".mp4")
        except VideoPrivate:
            print(li + " unavailable")


def download_from_row(line):
    try:
        if not os.path.exists("F:\\fyp\\cmcf_videos\\" + line[1] + ".mp4"):
            print(line[2])
            yt = YouTube(line[2])
            yt.streams.get_highest_resolution().download(
                "F:\\fyp\\cmcf_videos", line[1] + ".mp4")
    except VideoPrivate:
        print(" private")
    except VideoUnavailable:
        print(" unavailable")


def download_from_csv():
    with open("TODO") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            if row[0] == "oliver":
                download_from_row(row)


download_from_csv()
