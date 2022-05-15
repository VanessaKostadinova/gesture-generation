import cv2

input_folder = "F:\\fyp\\cut_data\\"
output_folder = "H:\\cut-data\\"

footage = cv2.VideoCapture(input_folder + "118_yq3TQoMjXTw.mp4")

c = 0
stop = False
while not stop:
    ret, frame = footage.read()
    if ret:
        cv2.imwrite(output_folder + "118_yq3TQoMjXTw" + "_" + f"{c:012d}" + ".jpg", frame)
        c += 1
    else:
        stop = True

footage.release()
