import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sequence_model import *
from dataset_creation import SequenceDataset
from data.render_poses import draw_pose
from matplotlib import animation
from viz import *

parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=15) #30 fps videos -> 15 = half second, 30 = second, ? = eight count
parser.add_argument('--dataroot', type=str, default='./data/dylan')
parser.add_argument('--statedir', type=str, default='./checkpoints') # where to load state dicts from
parser.add_argument('--load_state', type=str, default='latest') # which state dict to load; final corresponds to last save point
parser.add_argument('--savedir', type=str, default='./results')
parser.add_argument('--b', type=int, default=1) #this is a hack, dont change this
args = parser.parse_args()

if not os.path.exists(args.savedir):
    os.makedirs(args.savedir)

noise_dim = 64

generator = pose_generator(noise_dim)
generator.load_state_dict(torch.load(args.statedir + '/' + args.load_state + '_generator.pth'))
generator.eval()

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

generator.to(device)

seq_len = args.seq_len

i = 0
cont = True

dataset = SequenceDataset()
dataset.initialize(args)
loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.b)

with torch.no_grad():
    for b, real_seq in enumerate(loader):
        z = torch.empty(seq_len, 1, noise_dim).normal_(mean=0.0, std=1.0).to(device)
        start_and_end_poses = torch.cat([real_seq[:, 0], real_seq[:, 1]], dim=1).to(device)

        generated_seq = generator(z, start_and_end_poses).cpu().squeeze()

        animate_gan_sequence(generated_seq, args.savedir)
        animate_gan_sequence(real_seq[0], args.savedir)
