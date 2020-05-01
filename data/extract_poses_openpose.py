import cv2
import sys
import os
import glob
import pandas as pd

try:
	sys.path.append('/usr/local/python')
	from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

params = dict()
params["model_folder"] = "../../../models/"
params['hand'] = True
params['number_people_max'] = 1

def extract_poses(image_dir):
	print("Processing " + image_dir)
    # Read frames on directory
    imagePaths = op.get_images_on_directory(image_dir);
    start = time.time()

   	body = []
   	left_hand = []
   	right_hand = []

    # Process and display images
    for imagePath in imagePaths:
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        body.append(datum.poseKeypoints[0])
        left_hand.append(datum.handKeypoints[0][0])
        right_hand.append(datum.handKeypoints[1][0])

    end = time.time()
    print("OpenPose successfully finished on " + image_dir + ". Total time: " + str(end - start) + " seconds")

    df = pd.DataFrame({"body_points": body, "left_hand_points": left_hand, "right_hand_points": right_hand})
    df.to_csv(image_dir[:-4] + '_openpose.csv')

walk = list(os.walk('.'))
num_subjects = len(walk[0][1])
prev_offset = 1

# extract the frames
for i in range(num_subjects):
	for vid_name in walk[prev_offset][2]:
		if vid_name.split('.')[0] + "_frames" not in walk[prev_offset][1]:
			extract_frames(walk[prev_offset][0] + "/" + vid_name)
		else:
			print("Skipping " + vid_name)
	prev_offset += len(walk[prev_offset][1]) + 1
