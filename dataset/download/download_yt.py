from pytube import YouTube
from pytube.exceptions import VideoPrivate

url = "rgLgPTXdR4M"
name = "ringtone"

try:
    yt = YouTube('http://youtube.com/watch?v=' + url)
    yt.streams.order_by('resolution').desc().first().download(
        "C:\\Users\\vanes\\Downloads\\" + name + ".mp3")
except VideoPrivate:
    print(url + " unavailable")