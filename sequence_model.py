import torch
import torch.nn as nn

class pose_generator(nn.Module):
    def __init__(self, noise_dim):
        super(pose_generator, self).__init__()
        self.recurrent_block = nn.GRU(noise_dim, 100, 1)
        self.output_block = nn.Sequential(
            nn.ConvTranspose2d(2, 256, 4, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 3, 1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(inplace=True),
            )   

        #9 * 2 + 1 + 3 - 4 = 18
        # 17 * 2 + 1 + 3 - 6 = 32
        # 31*2 + 1 + 3 - 2 = 64 


        # (64 -7)//2 + 1 = 29
        # (29-5)//2 + 1 = 13
        #(13-4)//2 + 1 = 10
        self.condition = nn.Sequential(
            nn.Conv2d(6, 64, 7, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 5, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 1, 4),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, z, start_and_end_poses):
        T, B, _ = z.shape
        condition = self.condition(start_and_end_poses)
        print(condition.shape)
        out, _ = self.recurrent_block(z, condition.reshape(1, B, -1))

        out = out.reshape(T, B, 1, 10, 10)
        condition_tiled = condition.unsqueeze(0).repeat(15, 1, 1, 1, 1)
        out_c = torch.cat([out, condition_tiled], dim=2)

        poses = torch.empty(T, B, 3, 64, 64).double().to(z.device)
        for i in range(T):
            poses[i] = self.output_block(out_c[i])
        return poses

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(1, batch_size, 100).to(device).double()
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
        return torch.mean(validity, dim=1)

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
        return torch.mean(validity, dim=1)