from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

url_in = "F:\\fyp\\cmcf_videos\\Abortion_Laws_-_Last_Week_Tonight_with_John_Oliver_HBO-DRauXXz6t0Y.webm.mp4"
url_out = "C:\\Users\\vanes\\Desktop\\cmcf-test.mp4"

ffmpeg_extract_subclip(url_in,
                       float(58.6),
                       float(70.8),
                       url_out)
