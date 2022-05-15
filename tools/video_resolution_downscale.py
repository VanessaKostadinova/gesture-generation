import cv2

input_folder = "C:\\Users\\vanes\\Desktop\\"
output_folder = "C:\\Users\\vanes\\Desktop\\test\\"

resolution = (384, 216)

footage = cv2.VideoCapture(input_folder + "cmcf-test.mp4")

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

out = cv2.VideoWriter(output_folder + "cmcf.mp4", fourcc, 24, resolution)

stop = False
while not stop:
    ret, frame = footage.read()
    if ret:
        b = cv2.resize(frame, resolution, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        out.write(b)
    else:
        stop = True

footage.release()
out.release()
cv2.destroyAllWindows()