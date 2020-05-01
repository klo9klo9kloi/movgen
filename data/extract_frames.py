import cv2
import sys
import os
import glob

def extract_frames(filename):
	print("Processing " + filename)
	vidcap = cv2.VideoCapture(filename)
	success,image = vidcap.read()
	count = 0

	out_path = filename[:-4] + "_frames"
	if not os.path.exists(out_path):
		os.makedirs(out_path)

	while success:
	  cv2.imwrite(out_path + "/frame%d.jpg" % count, image)     # save frame as JPEG file
	  success,image = vidcap.read()
	  # print ('Read a new frame: ', success)
	  count += 1
	print("Done processing " + filename)

walk = list(os.walk('.'))
num_subjects = len(walk[0][1])
prev_offset = 1

for i in range(num_subjects):
	for vid_name in walk[prev_offset][2]:
		if vid_name.split('.')[0] + "_frames" not in walk[prev_offset][1]:
			extract_frames(walk[prev_offset][0] + "/" + vid_name)
		else:
			print("Skipping " + vid_name)
	prev_offset += len(walk[prev_offset][1]) + 1
