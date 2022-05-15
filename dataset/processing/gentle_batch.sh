#!/bin/bash

FILES="openpose_location/*"
VIDEOS="videos_folder"
OUTPUT_FOLDER="out_folder"
TRANSCRIPT_OUT="transcript_out_folder"

for f in $FILES
do
	echo "Processing $f file..."
	FILES=$(($(ls $f | wc -l)-1))
	NAME=$(basename "$f")
	echo $NAME
	if [ -f ${f}/transcript.txt ]; then
		cd ~/FYP/gentle-Final/

		if [ ! -f ${OUTPUT_FOLDER}${NAME}.wav ]; then
			ffmpeg -i ${VIDEOS}${NAME}.mp4 -ac 2 -f wav ${OUTPUT_FOLDER}${NAME}.wav
		fi
		
		if [ ! -f ${TRANSCRIPT_OUT}${NAME}.json ]; then
			python3 ./align.py ${OUTPUT_FOLDER}${NAME}.wav ${f}/transcript.txt --output ${TRANSCRIPT_OUT}${NAME}.json
		fi
	fi
done
