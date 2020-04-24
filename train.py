import torch
import torch.nn
import numpy as np
from models import PS_G, PS_D

# play with weighting the hand keypoints differently during training; but also since there are roughly double the hand keypoints as body keypoints, it maybe actually care more about
# fitting the hands, which is good theoretically but the hand data is super noisy so not sure what it would end up doing



#TODO: load data

print()
print('----------------------------------------')
print('Training with parameters: ' + str(self.get_params()))

pose_dim = ..
noise_dim = ..
hidden_dim = 50

generator = PS_G(noise_dim, pose_dim, pose_dim, hidden_dim)
discriminator = PS_D(pose_dim, pose_dim, hidden_dim)

all_training_data = torch.from_numpy(X.astype('float64'))
all_training_labels = torch.from_numpy(y.astype('float64'))
data = TensorDataset(all_training_data, all_training_labels)
loader = DataLoader(data, shuffle=True, batch_size=32)

is_cuda = torch.cuda.is_available()
DoubleTensor = torch.cuda.DoubleTensor if is_cuda else torch.DoubleTensor

if is_cuda:
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

generator.double()
generator.to(device)

discriminator.double()
discriminator.to(device)

lr = 1e-3
weight_decay = 1e-5

#set up loss function and optimizer
g_optim = torch.optim.RMSProp(generator.parameters(), lr=lr, weight_decay=weight_decay)
d_optim = torch.optim.RMSProp(discriminator.parameters(), lr=lr, weight_decay=weight_decay)

epochs = 100
seq_len = 15 #30 fps videos -> 15 = half second, 30 = second, ? = eight count
generator_update_freq = 5

# train
for i in range(epochs):
	for b, (true_poses, start_poses) in enumerate(loader):
		valid = Variable(DoubleTensor(inpts.size(0), 1).fill_(1.0), requires_grad=False) #implicit .to(device) because of earlier class declaration
		fake = Variable(DoubleTensor(inpts.size(0), 1).fill_(0.0), requires_grad=False)
		'''
		Train Discriminator
		'''
		d_optim.zero_grad()
		
		z = Variable(DoubleTensor(np.random.normal(0, 1, (seq_len, inpts.size(0), noise_dim))))

		true_poses = true_poses.to(device)
		start_poses = start_poses.to(device)
		g_hidden = generator.init_hidden(true_poses.size(0)).to(device)
		d_hidden = discriminator.init_hidden(true_poses.size(0)).to(device)

		generated_poses = generator(z, start_poses, g_hidden)
		d_loss = -torch.mean(discriminator(generated_poses, start_poses, d_hidden)) + torch.mean(discriminator(true_poses, start_poses, d_hidden))

		d_loss.backward()
		d_optim.step()
		# clip weights to a fixed threshold
		for p in discriminator.parameters():
    		p.data.clamp_(-0.01, 0.01)

		if b % generator_update_freq == 0:

			''' 
			Train Generator
			'''
			g_optim.zero_grad()

			generated_poses = generator(z, start_poses, g_hidden)
			g_loss = -torch.mean(discriminator(generated_poses, start_poses, d_hidden))

			g_loss.backward()
			g_optim.step()

		print(
        	"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        	% (i, epochs, b, len(loader), d_loss.item(), g_loss.item())
    	)

torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')