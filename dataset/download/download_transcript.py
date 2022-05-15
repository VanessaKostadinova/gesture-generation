import json
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from tools.string_tools import sanitise_filesystem_name

id_file = open("F:\\fyp\\video_ids.txt", "r")
id_list = id_file.readlines()

transcript_path = "F:\\fyp\\transcripts\\"


for link in id_list:
    json_file_path = transcript_path + sanitise_filesystem_name(link) + ".json"
    if not os.path.exists(json_file_path):
        try:
            print(json_file_path)
            transcript = YouTubeTranscriptApi.get_transcript(link)
            out = open(json_file_path, "x")
            json.dump(transcript, out)
            out.close()
        except TranscriptsDisabled:
            print(sanitise_filesystem_name(link) + " not available")