import cv2
import numpy as np
#import pathlib
#import pandas as pd
import json
import time
#import logging
import math


vidoefile = '/home/mingming/Downloads/0548207774494477-20190501-down-163849.mkv'

cap = cv2.VideoCapture(vidoefile)
video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print (video_len)

jsfile = vidoefile.replace("mkv","json")

frame_list = []

with open(jsfile) as f:
    for row in f.readlines():
        #lines.append(row)
        data = json.loads(row)
        frames = data["frames"]
        frame_list = frame_list + frames

json_len = len(frame_list)

print("frame in json: ",len(frame_list))