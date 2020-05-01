import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from sequence_model import PS_G, PS_D
from dataset_creation import SequenceDataset

def save_network(network, network_label, epoch_label, save_dir):
	save_filename = '%s_%s.pth' % (epoch_label, network_label)
	save_path = os.path.join(save_dir, save_filename)
	torch.save(network.cpu().state_dict(), save_path)
	if torch.cuda.is_available():
		network.cuda()

def load_network(network_label, epoch_label, save_dir):
	load_filename = '%s_%s.pth' % (epoch_label, network_label)
	load_path = os.path.join(save_dir, load_filename)
	return torch.load(load_path)

# play with weighting the hand keypoints differently during training; but also since there are roughly double the hand keypoints as body keypoints, it maybe actually care more about
# fitting the hands, which is good theoretically but the hand data is super noisy so not sure what it would end up doing

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--b', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seq_len', type=int, default=15) #30 fps videos -> 15 = half second, 30 = second, ? = eight count
parser.add_argument('--g_update_freq', type=int, default=5)
parser.add_argument('--dataroot', type=str, default='./data/dylan')
parser.add_argument('--savedir', type=str, default='./checkpoints')
parser.add_argument('--save_epoch_freq', type=int, default=5)
parser.add_argument('--use_confidence', action='store_true', default=False)
parser.add_argument('--clip_value', type=float, default=0.01)
parser.add_argument('--continue_from', type=int, default=0)
args = parser.parse_args()

if not os.path.exists(args.savedir):
	os.makedirs(args.savedir)

n_points = 25 + 21 + 21
pose_dim = 2*n_points # if we ignore confidence score
# pose_dim = 3 * n_points # but confidence might be helpful in ignoring bad predictions, and we cant filter cause we need a common dimension across entire dataset

generator = PS_G(pose_dim, args.hidden_dim).double()
discriminator = PS_D(pose_dim, args.hidden_dim).double()

if args.continue_from > 0:
	generator.load_state_dict(load_network('generator', args.continue_from, args.savedir))
	discriminator.load_state_dict(load_network('discriminator', args.continue_from, args.savedir))

dataset = SequenceDataset()
dataset.initialize(args)
loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.b)

is_cuda = torch.cuda.is_available()

if is_cuda:
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

generator.to(device)
discriminator.to(device)

#set up optimizers
g_optim = torch.optim.RMSprop(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
d_optim = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)

seq_len = args.seq_len
generator_update_freq = args.g_update_freq

# train
for i in range(args.continue_from, args.continue_from + args.epochs):
	for b, (start_seq, end_seq, weights) in enumerate(loader):
		assert(start_seq.size(2) == pose_dim)
		'''
		Train Discriminator
		'''
		d_optim.zero_grad()
		discriminator.zero_grad()

		start_seq = start_seq.to(device).double().transpose(1,0)
		end_seq = end_seq.to(device).double().transpose(1,0)

		final_pose = end_seq[-1, :, :]
		p_T_chain = final_pose.unsqueeze(0).repeat(seq_len, 1, 1).to(device).double()

		g_hidden = generator.init_hidden(start_seq.size(1), device)
		d_hidden = discriminator.init_hidden(start_seq.size(1), device)

		generated_seq = generator(start_seq, p_T_chain, g_hidden).detach()
		# print(discriminator(end_seq, p_T_chain, d_hidden).shape)
		d_real = torch.mean(discriminator(end_seq, p_T_chain, d_hidden))
		d_fake = torch.mean(discriminator(generated_seq, p_T_chain, d_hidden))
		# d_real = torch.mean(discriminator(end_seq, final_pose))
		# d_fake = torch.mean(discriminator(generated_seq, final_pose))
		d_loss = -(d_real - d_fake)

		d_loss.backward()
		d_optim.step()

		# clip weights to a fixed threshold
		for p in discriminator.parameters():
			p.data.clamp_(-args.clip_value, args.clip_value)

		if b % generator_update_freq == 0:
			''' 
			Train Generator
			'''
			g_optim.zero_grad()
			generator.zero_grad()
			
			generated_seq = generator(start_seq, p_T_chain, g_hidden)
			d_real_g = -torch.mean(discriminator(generated_seq, p_T_chain, d_hidden))
			# d_real_g = -torch.mean(discriminator(generated_seq, final_pose))

			coord_difference = torch.pow(generated_seq - end_seq, 2)
			weights = weights.to(device).double().transpose(1, 0)
			point_MSE = torch.mean(weights * (coord_difference[:, :, np.arange(0, pose_dim, 2)] + coord_difference[:, :, np.arange(1, pose_dim, 2)]))

			g_loss = d_real_g + point_MSE

			g_loss.backward()
			g_optim.step()

		print(
			"[Epoch %d/%d] [Batch %d/%d] [D fake: %f] [D real: %f] [G loss: %f] [PMSE: %f]"
			% (i, args.epochs + args.continue_from, b, len(loader), d_fake.item(), d_real.item(), d_real_g.item(), point_MSE.item())
		)
	if i % args.save_epoch_freq == 0:
		save_network(generator, 'generator', i, args.savedir)
		save_network(discriminator, 'discriminator', i, args.savedir)

save_network(generator, 'generator', args.epochs + args.continue_from, args.savedir)
save_network(discriminator, 'discriminator', args.epochs + args.continue_from, args.savedir)
save_network(generator, 'generator', 'latest', args.savedir)
save_network(discriminator, 'discriminator', 'latest', args.savedir)