import moviepy.editor as mp
import os
import time

videos_path = "/home/nvr/NVRstorage/Recordings/2/main/"
audio_path = "/home/nvr/converted_audio_files/"
video_file = ''
prev_file = video_file

while True:
    current = os.popen('ls {}'.format(audio_path)).read()
    if len(current) == 0:
        video_file = os.popen("ls {} | tail -n 2 | head -n 1".format(videos_path)).read()[:-1]
        if prev_file == video_file:
            time.sleep(0.2)
            continue
        prev_file = video_file
        vid = mp.VideoFileClip(videos_path + video_file)
        vid.audio.write_audiofile(audio_path + video_file.split('.')[0] + '.wav')
