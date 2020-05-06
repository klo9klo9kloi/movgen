import torch
import torch.nn as nn

class pose_generator(nn.Module):
    def __init__(self, noise_dim, n_layers=1):
        super(pose_generator, self).__init__()
        self.recurrent_block = nn.GRU(noise_dim, 256, n_layers)
        self.output_block = nn.Sequential(
            nn.ConvTranspose2d(1, 256, 6),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=4),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2, padding=4),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 3, 1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(inplace=True),
            )

        self.n_layers = n_layers

    def forward(self, z, hidden):
        T, B, _ = z.shape
        out, _ = self.recurrent_block(z, hidden)
        D = int(out.size(-1)**(1/2))
        out = out.reshape(T, B, 1, D, D)
        poses = torch.empty(T, B, 3, 64, 64).double().to(z.device)
        for i in range(T):
            poses[i] = self.output_block(out[i])
        return poses

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.n_layers, batch_size, 256).to(device).double()
        return hidden

class frame_discriminator(nn.Module):
    def __init__(self):
        super(frame_discriminator, self).__init__()

        self.output_block = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, sample):
        N = sample.size(0)
        validity = self.output_block(sample).reshape(N, -1)
        return validity

class sequence_discriminator(nn.Module):
    def __init__(self, n_layers=1):
        super(sequence_discriminator, self).__init__()

        self.output_block = nn.Sequential(
            nn.Conv3d(3, 64, 4),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 128, 4),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 256, 4),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(256, 1, 4),
            nn.Sigmoid()
        )

        self.n_layers = n_layers

    def forward(self, sample):
        # sample will have shape (T, N, C, H, W), need to convert to (N, C, T, H, W)
        T, N, C, H, W = sample.shape
        reshaped = torch.empty(N, C, T, H, W)
        for i in range(T):
            reshaped[:, :, i, :, :] = sample[i]
        validity = self.output_block(reshaped.to(sample.device).double()).reshape(N, -1)
        return validity