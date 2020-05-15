import os
import cv2
import matplotlib.cm as cm
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import skimage.io as skio

def denormalize(points, D=640):
	return (0.5 * ((points + 1.0) * D))

def draw_pose_alphapose(kps, H, W):
	"""
	0: Right heel
	1: Right knee
	2: Right hip
	3: Left hip
	4: Left knee
	5: Left heel
	6: Right wrist
	7: Right elbow
	8: Right shoulder
	9: Left shoulder
	10: Left elbow
	11: Left wrist
	12: Neck
	13: Head top
	14: nose
	15: left_eye
	16: right_eye
	17: left_ear
	18: right_ear
	19: left big toe
	20: right big toe
	21: Left small toe
	22: Right small toe
	23: L ankle
	24: R ankle
	"""
	body_part_pairs = [ 0, 24, 1, 2, 2, 8, 3, 9, 4, 3, 5,23, 6,7, 7,8, 8,12, 9,12, 10, 9, 11,10, 12,14, 14,13, 17,15,
			18,16, 19,23, 20,24, 21,23, 22,24, 23,4, 24,1, 2,3]
	kps = denormalize(kps)
	img = np.zeros((H, W, 3), dtype=np.uint8)
	body_colors = cm.rainbow(np.linspace(0, 1, len(body_part_pairs)))
	for i in range(0, len(body_part_pairs), 2):
		p1 = (int(kps[body_part_pairs[i]*2]), int(kps[body_part_pairs[i]*2 + 1]))
		p2 = (int(kps[body_part_pairs[i+1]*2]), int(kps[body_part_pairs[i+1]*2 + 1]))
		color = tuple(body_colors[i][:3] * 255)
		img = cv2.line(img, p1, p2, color, 5)
		img = cv2.circle(img, p2, 6, color, -1)
	return img

def draw_pose_openpose(kps, H, W):
	# replace 13 with midhip estimate
	body_part_pairs = [12,13,   12,8, 8,7, 7,6,   12,9, 9,10, 10,11,   13,2, 2,1, 1,24,   13,3, 3,4, 4,23,   
				12,14, 14,16, 16,18, 14,15, 15,17, 23,19,19,21,23,5, 24,20,20,22,24,0]
	kps[2*13:2*13+2] = (kps[2*3:2*3+2] + kps[2*2:2*2+2])/2
	kps = denormalize(kps)
	img = np.zeros((H, W, 3), dtype=np.uint8)
	body_colors = cm.rainbow(np.linspace(0, 1, len(body_part_pairs)))
	for i in range(0, len(body_part_pairs), 2):
		p1 = (int(kps[body_part_pairs[i]*2]), int(kps[body_part_pairs[i]*2 + 1]))
		p2 = (int(kps[body_part_pairs[i+1]*2]), int(kps[body_part_pairs[i+1]*2 + 1]))
		color = tuple(body_colors[i][:3] * 255)
		img = cv2.line(img, p1, p2, color, 5)
		img = cv2.circle(img, p2, 6, color, -1)
	return img

def animate_sequence(sequence, savedir, save_images=False):
	if not os.path.exists(savedir):
		os.makedirs(savedir)
	poses = []
	fig = plt.figure()
	plt.axis('off')
	im = plt.imshow(np.zeros((640, 640, 3))) #parametrize img shape
	def init():
		im.set_data(poses[0])
		return [im]
	def animate(i):
		im.set_data(poses[i])
		return [im]
	n = sequence.shape[0]
	poses = [draw_pose_openpose(sequence[i].flatten(), 640, 640) for i in range(n)]
	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(poses), interval=33.3, blit=True) #parametrize fps
	anim.save(os.path.join(savedir, 'anim.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	plt.show()

	if save_images:
		for i in range(len(poses)):
			skio.imsave(os.path.join(savedir, str(i)+'.png'), poses[i])

def animate_gan_sequence(sequence, savepath, G_dim=64):
	fig = plt.figure()
	im = plt.imshow(np.zeros((G_dim, G_dim, 3))) #parametrize img shape
	sequence = np.clip(sequence, 0, 1)
	def init():
		t = np.zeros((G_dim, G_dim, 3))
		t[:, :, 0] = sequence[0][0]
		t[:, :, 1] = sequence[0][1]
		t[:, :, 2] = sequence[0][2]
		im.set_data(t)
		return [im]
	def animate(i):
		t = np.zeros((G_dim, G_dim, 3))
		t[:, :, 0] = sequence[i][0]
		t[:, :, 1] = sequence[i][1]
		t[:, :, 2] = sequence[i][2]
		im.set_data(t)
		return [im]
	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(sequence), interval=33.3, blit=True) #parametrize fps
	anim.save(savepath, fps=30, extra_args=['-vcodec', 'libx264'])
	plt.show()