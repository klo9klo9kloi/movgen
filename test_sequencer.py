import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sequence_model import PS_G, PS_D
from dataset_creation import SequenceDataset
from data.render_poses import draw_pose
from matplotlib import animation

parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=15) #30 fps videos -> 15 = half second, 30 = second, ? = eight count
parser.add_argument('--dataroot', type=str, default='./data/dylan')
parser.add_argument('--statedir', type=str, default='./checkpoints') # where to load state dicts from
parser.add_argument('--load_state', type=str, default='latest') # which state dict to load; final corresponds to last save point
parser.add_argument('--savedir', type=str, default='./results')
parser.add_argument('--b', type=int, default=1) #this is a hack, dont change this
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--use_confidence', action='store_true', default=False)
args = parser.parse_args()

if not os.path.exists(args.savedir):
	os.makedirs(args.savedir)

######## FOR ANIMATION

poses = []
def init():
	im.set_data(poses[0])
	return [im]

def animate(i):
	im.set_data(poses[i])
	return [im]

pose_dim = 2*(25 + 21 + 21) # if we ignore confidence score
# pose_dim = 75 + 63 + 63 # but confidence might be helpful in ignoring bad predictions, and we cant filter cause we need a common dimension across entire dataset

generator = PS_G(pose_dim, args.hidden_dim).double()
# print(args.statedir + '/' + args.load_state + '_generator.pth')
generator.load_state_dict(torch.load(args.statedir + '/' + args.load_state + '_generator.pth'))
generator.eval()

dataset = SequenceDataset()
dataset.initialize(args)

is_cuda = torch.cuda.is_available()

if is_cuda:
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

generator.to(device)

seq_len = args.seq_len

i = 0
cont = True

while cont:
	with torch.no_grad():
		index = np.random.randint(len(dataset))
		feed, truth, _ = dataset[index]
		print(feed)

		truth = truth.unsqueeze(0).transpose(1, 0)
		feed = feed.unsqueeze(0).to(device).double().transpose(1, 0)

		p_T_chain = truth[-1, :, :].unsqueeze(0)
		p_T_chain = p_T_chain.repeat(seq_len, 1, 1).to(device).double()

		g_hidden = generator.init_hidden(1, device)

		generated_seq = generator(feed, p_T_chain, g_hidden).squeeze().data.cpu().numpy()
		truth = truth.squeeze().cpu().numpy()
		if not args.use_confidence:
			indxs = np.arange(2, pose_dim+1, 2)
			generated_seq = np.insert(generated_seq, indxs, 1, axis=1)
			truth = np.insert(truth, indxs, 1, axis=1)

		fig = plt.figure()
		im = plt.imshow(np.zeros((640, 640, 3))) #parametrize img shape
		print(generated_seq)
		poses = [draw_pose(keypoints) for keypoints in generated_seq]
		anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(poses), interval=30, blit=True) #parametrize fps
		anim.save(args.savedir + '_fake_' + str(i)  + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

		fig = plt.figure()
		im = plt.imshow(np.zeros((640, 640, 3))) #parametrize img shape
		poses = [draw_pose(keypoints) for keypoints in truth]
		print(truth)
		anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(poses), interval=30, blit=True) #parametrize fps
		anim.save(args.savedir + '_real_' + str(i)  + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

	cont = input('Press y to continue, any other key to stop. ') == 'y'
