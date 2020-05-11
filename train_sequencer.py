import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import skimage.io as skio
from sequence_model import *
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
parser.add_argument('--b', type=int, default=32)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seq_len', type=int, default=15) #30 fps videos -> 15 = half second, 30 = second, ? = eight count
# parser.add_argument('--g_update_freq', type=int, default=5)
# parser.add_argument('--d_update_freq', type=int, default=5)
parser.add_argument('--switch_freq', type=int, default=1)
parser.add_argument('--dataroot', type=str, default='./data/dylan')
parser.add_argument('--savedir', type=str, default='./checkpoints')
parser.add_argument('--save_epoch_freq', type=int, default=10)
parser.add_argument('--use_confidence', action='store_true', default=False)
parser.add_argument('--clip_value', type=float, default=0.01)
parser.add_argument('--continue_from', type=int, default=0)
args = parser.parse_args()

if not os.path.exists(args.savedir):
    os.makedirs(args.savedir)

noise_dim = 64

generator = pose_generator(noise_dim)
discriminator_f = frame_discriminator()
discriminator_s = sequence_discriminator()
discriminator_start = frame_discriminator()
disciminator_end = frame_discriminator()

# if args.continue_from > 0:
#     generator.load_state_dict(load_network('generator', args.continue_from, args.savedir))
#     discriminator_f.load_state_dict(load_network('discriminator_f', args.continue_from, args.savedir))
#     discriminator_s.load_state_dict(load_network('discriminator_s', args.continue_from, args.savedir))
# elif args.continue_from < 0:
#     generator.load_state_dict(load_network('generator', "latest", args.savedir))
#     discriminator_f.load_state_dict(load_network('discriminator_f', "latest", args.savedir))
#     discriminator_s.load_state_dict(load_network('discriminator_s', "latest", args.savedir))
#     args.continue_from = 0

dataset = SequenceDataset()
dataset.initialize(args)
loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.b)

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
generator.to(device)
discriminator_f.to(device)
discriminator_s.to(device)
disciminator_start.to(device)
disciminator_end.to(device)

#set up optimizers
g_optim = torch.optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))
d_f_optim = torch.optim.Adam(discriminator_f.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))
d_s_optim = torch.optim.Adam(discriminator_s.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))
d_start_optim = torch.optim.Adam(discriminator_start.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))
d_end_optim = torch.optim.Adam(discriminator_end.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))

seq_len = args.seq_len
generator_update_freq = args.g_update_freq
discriminator_update_freq = args.d_update_freq

criterion = nn.BCELoss()

turn = 0

# train
for i in range(args.continue_from, args.continue_from + args.epochs):
    if i % args.switch_freq == 0:
        turn = 1 - turn

    # train generator
    if turn:
        discriminator_s.eval()
        discriminator_f.eval()
        discriminator_start.eval()
        discriminator_end.eval()
        generator.train()

        for b, real_seq in enumerate(loader):
            batch_size = real_seq.size(0)

            valid = torch.ones((batch_size, 1), requires_grad=False).to(device)
            fake = torch.zeros((batch_size, 1), requires_grad=False).to(device)

            f_index = np.random.randint(0, seq_len)

            g_optim.zero_grad()
            
            z = torch.empty(seq_len, batch_size, noise_dim).normal_(mean=0.0, std=1.0).to(device)
            start_and_end_poses = torch.cat([real_seq[:, 0], real_seq[:, 1]], dim=1).to(device)

            generated_seq = generator(z, start_and_end_poses)

            g_loss = criterion(discriminator_f(generated_seq[f_index]), valid)
            g_loss += criterion(discriminator_start(generated_seq[0]), valid)
            g_loss += criterion(discriminator_end(generated_seq[1]), valid)
            g_loss += criterion(discriminator_s(generated_seq), valid)

            g_loss.backward()
            g_optim.step()  

            print(
                "[Epoch %d/%d] [Batch %d/%d] [G loss: %f] "
                % (i, args.epochs + args.continue_from, b, len(loader), g_loss.item())
            
            )

    # train discminator
    else:
        discriminator_s.train()
        discriminator_f.train()
        discriminator_start.train()
        discriminator_end.train()
        generator.eval()

        for b, real_seq in enumerate(loader):
            batch_size = real_seq.size(0)

            valid = torch.ones((batch_size, 1), requires_grad=False).to(device)
            fake = torch.zeros((batch_size, 1), requires_grad=False).to(device)

            f_index = np.random.randint(0, seq_len)

            d_f_optim.zero_grad()
            d_s_optim.zero_grad()
            d_start_optim.zero_grad()
            d_end_optim.zero_grad()

            z = torch.empty(seq_len, batch_size, noise_dim).normal_(mean=0.0, std=1.0).to(device)
            start_and_end_poses = torch.cat([real_seq[:, 0], real_seq[:, 1]], dim=1).to(device)
            generated_seq = generator(z, start_and_end_poses)

            real_seq = real_seq.to(device).double().transpose(1, 0)

            d_real = criterion(discriminator_f(real_seq[f_index]), valid)
            d_real += criterion(disciminator_start(real_seq[0]), valid)
            d_real += criterion(discriminator_end(real_seq[-1]), valid)
            d_real += criterion(discriminator_s(real_seq), valid)

            d_fake = criterion(discriminator_f(generated_seq[f_index]), fake)
            d_fake += criterion(disciminator_start(generated_seq[0]), fake)
            d_fake += criterion(discriminator_end(generated_seq[0]), fake)
            d_fake += criterion(disciminator_s(generated_seq), fake)

            d_loss = 0.5 * (d_real + d_fake)

            d_loss.backward()
            d_f_optim.step()
            d_s_optim.step()
            d_start_optim.step()
            d_end_optim.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] "
                % (i, args.epochs + args.continue_from, b, len(loader), d_loss.item())
            )
    if (i-1) % args.save_epoch_freq == 0:
        save_network(generator, 'generator', i, args.savedir)
        # save_network(discriminator_f, 'discriminator_f', i, args.savedir)
        # save_network(discriminator_s, 'discriminator_s', i, args.savedir)

save_network(generator, 'generator', args.epochs + args.continue_from, args.savedir)
# save_network(discriminator_f, 'discriminator_f', args.epochs + args.continue_from, args.savedir)
# save_network(discriminator_s, 'discriminator_s', args.epochs + args.continue_from, args.savedir)
# save_network(generator, 'generator', 'latest', args.savedir)
# save_network(discriminator_f, 'discriminator_f', 'latest', args.savedir)
# save_network(discriminator_s, 'discriminator_s', 'latest', args.savedir)