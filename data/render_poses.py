import os
import sys
import json
import glob
import numpy as np
import skimage.io as skio
import cv2
import matplotlib.cm as cm

H135 = 70

# 24 colors; roygbv
body_part_pairs = [1,8,   1,2, 2,3, 3,4,   1,5, 5,6, 6,7,   8,9, 9,10, 10,11,   8,12, 12,13, 13,14,   1,0, 0,15, 15,17, 0,16, 16,18, 14,19,19,20,14,21, 11,22,22,23,11,24]
left_hand_pairs = [
              -1,0, 0,1, 1,2, 2,3, 3,4, -1,5, 5,6, 6,7,
               7,8, -1,9, 9,10, 10,11, 11,12, -1,13, 13,14, 14,15,
               15,16, -1,17, 17,18, 18,19, 19,20
               ]
right_hand_pairs = [
               -1,0, 0,1, 1,2, 2,3, 3,4, -1,5, 5,6, 6,7,
               7,8, -1,9, 9,10, 10,11, 11,12, -1,13, 13,14, 14,15,
               15,16, -1,17, 17,18, 18,19, 19,20
]

def draw_pose(full_keypoints_list):
	img = np.zeros((640, 640, 3), dtype=np.uint8)
	body_colors = cm.rainbow(np.linspace(0, 1, len(body_part_pairs)))
	hand_colors = cm.rainbow(np.linspace(0, 1, len(left_hand_pairs)))

	#draw body points
	body_points = full_keypoints_list[:25 * 3]
	for i in range(0, len(body_part_pairs), 2):
		if body_points[body_part_pairs[i]*3 + 2] > 0.5: # only accept confident predictions
			p1 = (int(body_points[body_part_pairs[i]*3]), int(body_points[body_part_pairs[i]*3 + 1]))
			p2 = (int(body_points[body_part_pairs[i+1]*3]), int(body_points[body_part_pairs[i+1]*3 + 1]))

			color = tuple(body_colors[i][:3] * 255)
			img = cv2.line(img, p1, p2, color, 5)
			img = cv2.circle(img, p2, 6, color, -1)

	# draw left hand
	left_hand_points = full_keypoints_list[25*3: 25*3+21*3]
	left_wrist = (int(body_points[7*3]), int(body_points[7*3 + 1]))
	for i in range(0, len(left_hand_pairs), 2):
		if left_hand_points[left_hand_pairs[i]*3 + 2] > 0.5: # only accept confident predictions
			if left_hand_pairs[i] < 0:
				p1 = left_wrist
			else:
				p1 = (int(left_hand_points[left_hand_pairs[i]*3]), int(left_hand_points[left_hand_pairs[i]*3 + 1]))
			p2 = (int(left_hand_points[left_hand_pairs[i+1]*3]), int(left_hand_points[left_hand_pairs[i+1]*3 + 1]))

			color = tuple(hand_colors[i][:3] * 255)
			img = cv2.line(img, p1, p2, color, 3)
			img = cv2.circle(img, p2, 4, color, -1)


	# draw right hand
	right_hand_points = full_keypoints_list[25*3 + 21*3:]
	right_wrist = (int(body_points[4*3]), int(body_points[4*3 + 1]))
	for i in range(0, len(right_hand_pairs), 2):
		if right_hand_points[right_hand_pairs[i]*3 + 2] > 0.5: # only accept confident predictions
			if right_hand_pairs[i] < 0:
				p1 = right_wrist
			else:
				p1 = (int(right_hand_points[right_hand_pairs[i]*3]), int(right_hand_points[right_hand_pairs[i]*3 + 1]))
			p2 = (int(right_hand_points[right_hand_pairs[i+1]*3]), int(right_hand_points[right_hand_pairs[i+1]*3 + 1]))

			color = tuple(hand_colors[i][:3] * 255)
			img = cv2.line(img, p1, p2, color, 3)
			img = cv2.circle(img, p2, 4, color, -1)
	return img

def main():
	args = sys.argv[1:]
	assert(len(args) == 1)
	path = args[0] + '/keypoints/'
	json_paths = glob.glob(path + "*.json")
	save_dir = args[0] + '/poses/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	for frame_json in json_paths:
		with open(frame_json, 'rb') as f:
			j = json.load(f)
			# print(frame_json)

			img = draw_pose(j['people'][0]['pose_keypoints_2d'] + j['people'][0]['hand_left_keypoints_2d'] + j['people'][0]['hand_right_keypoints_2d'])

			# skio.imshow(img, cmap='gray')
			# skio.show()
			skio.imsave(save_dir + frame_json.split('/')[-1].split('.')[0] + '.jpg', img)

if __name__ == '__main__':
	main()

