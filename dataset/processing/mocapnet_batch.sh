#!/bin/bash

FILES="file_location/*"
ZEROES=12

for f in $FILES
do
	echo "Processing $f file..."
	FILES=$(($(ls $f | wc -l)-1))
	NAME=$(basename "$f")
	if [ ! -f ${f}/2dJoints_v1.4.csv ]; then
		~/FYP/Hand/MocapNET/convertOpenPoseJSONToCSV --from $f --seriallength $ZEROES --label ${NAME}_ --startAt 0 --maxFrames $FILES --size 256 128

	fi
	if [ ! -f ${f}/arm.bvh ]; then
		cd ~/FYP/Arm/MocapNET/

		sudo ./MocapNET2CSV --from ${f}/2dJoints_v1.4.csv --ik 0.01 15 40 --novisualization --${NAME}_ --seriallength $ZEROES --size 256 128 --nolowerbody

		mv ./out.bvh ${f}/arm.bvh
	fi
	if [ ! -f ${f}/hand.bvh ]; then
		cd ~/FYP/Hand-2/MocapNET/

		sudo ./MocapNET2CSV --from ${f}/2dJoints_v1.4.csv --ik 0.01 15 40 --novisualization --hands --${NAME}_ --seriallength $ZEROES --size 256 128 --nolowerbody

		mv ./out.bvh ${f}/hand.bvh
	fi
done
