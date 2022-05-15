import json
import os
import csv

bvh_path = "F:\\fyp\\cut_data_openpose\\"
csv_path = "F:\\fyp\\filtered_data\\"
transcript_path = "F:\\fyp\\transcripts\\"


def get_first_index_of_possible_candidates(transcript, start_time):
    for i in range(0, len(transcript)):
        if (transcript[i]["start"]) >= start_time:
            if i != 0:
                return i - 1
            else:
                return i
    return None


def get_relevant_transcript():
    for f in os.listdir(bvh_path):
        name = f.split("_", 1)[1]
        scene = f.split("_", 1)[0]

        reader = csv.reader(open(csv_path + name + ".csv", "r"))

        for i in range(0, int(scene) + 1):
            reader.__next__()

        line = reader.__next__()

        start_time = float(line[3])
        print(start_time)
        duration_time = float(line[4])

        # if the transcript exists
        if os.path.exists(transcript_path + name + ".json"):
            # load
            transcript_file = json.load(open(transcript_path + name + ".json", "r", encoding='utf8'))
            candidate_index = get_first_index_of_possible_candidates(transcript_file, start_time)

            if candidate_index is None:
                continue

            # check if first possible candidate is valid
            if transcript_file[candidate_index]["start"] + transcript_file[candidate_index]["duration"] >= start_time:
                first_index = candidate_index
            else:
                first_index = candidate_index + 1

            last_index = len(transcript_file)

            # keep going until we reach word out of scope
            for i in range(first_index, last_index):
                if float(transcript_file[i]["start"]) > start_time + duration_time:
                    last_index = i
                    break

            # put everything together and clean text
            phrase = ""
            for i in range(first_index, last_index):
                text = transcript_file[i]["text"]
                text = list(text)

                for c in range(0, len(text)):
                    if text[c] == "\\":
                        text[c] = ""
                        if not c == len(text) & text[c + 1] == "n":
                            text[c + 1] = " "
                    phrase += text[c]

                phrase += " "

            if not os.path.exists(bvh_path + f + "\\transcript.txt"):
                open_type = "x"
            else:
                open_type = "w"

            with open(bvh_path + f + "\\transcript.txt", open_type, encoding='utf8', errors="ignore") as out:
                out.write(phrase)


get_relevant_transcript()
