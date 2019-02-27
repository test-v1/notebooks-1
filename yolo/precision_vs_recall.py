import numpy as np
import pandas as pd
import sys
import cv2
import glob, os
import itertools

import classifier

file_list_bad = sorted(glob.glob('validation/bad/*.jpg'))
file_list_good = sorted(glob.glob('validation/good/*.jpg'))

if len(sys.argv) > 4:
  classifier.set_gpu(int(sys.argv[4]))

netMain, metaMain = classifier.load_net(sys.argv[1], sys.argv[3], sys.argv[2])
detections_good = [classifier.detect(netMain, metaMain, cv2.imread(file), thresh=0.25) for file in file_list_good]
detections_bad = [classifier.detect(netMain, metaMain, cv2.imread(file), thresh=0.25) for file in file_list_bad]

good_detected = len([ d for d in detections_good if len(d) > 0])
bad_detected = len([ d for d in detections_bad if len(d) > 0])

precision = float(bad_detected) / (bad_detected + good_detected)
recall = float(bad_detected) / len(file_list_bad)

print(sys.argv[3], precision, recall)
