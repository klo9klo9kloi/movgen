import torch
import torch.nn

#######################################################################################
#                                 Pose Sequence Modeling                              #
#######################################################################################


class PS_G(nn.Module):
    def __init__(self, noise_dim, label_dim, output_dim, hidden_dim_base, n_layers=1):
        super(CGAN_Generator, self).__init__()

        self.noise_block = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim_base),
            nn.BatchNorm1d(hidden_dim_base, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
            )
        # self.label_block = nn.Sequential(      for using raw image instead of pose label
        #     nn.Conv2d(),
        #     nn.BatchNorm2d(),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(),
        #     nn.BatchNorm2d(),
        #     nn.LeakyReLU(),
        #     )
        self.label_block = nn.Sequential(
            nn.Linear(label_dim, hidden_dim_base),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.lstm = nn.LSTM(2*hidden_dim_base, 2*hidden_dim_base, n_layers)
        self.output_block = nn.Sequential(
            nn.Linear(2*hidden_dim_base, output_dim),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.noise_dim = noise_dim
        self.n_layers = n_layers
        self.hidden_dim_base

    def forward(self, z, y, hidden):
        # y shape (T, B, D)
        z = self.noise_block(z)
        y = self.label_block(y)

        out = torch.cat([z, y], dim=2)
        out, _ = self.lstm(out, hidden) # shape (T, B, 2H)
        out = self.output_block(out)
        return out

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers, batch_size, 2*self.hidden_dim_base).double(),
                      torch.zeros(self.n_layers, batch_size, 2*self.hidden_dim_base).double())
        return hidden

class PS_D(nn.Module):
    def __init__(self, sample_dim, label_dim, hidden_dim_base, n_layers=1):
        super(CGAN_Discriminator, self).__init__()

        def block(in_dim, out_dim):
            layers = [nn.Linear(in_dim, out_dim)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.sample_block = nn.Sequential(
            *block(sample_dim, hidden_dim_base)
            )
        self.label_block = nn.Sequential( 
            *block(label_dim, hidden_dim_base)
            )

        self.lstm = nn.LSTM(2*hidden_dim_base, 2*hidden_dim_base, n_layers)
        self.output_block = nn.Sequential(
            *block(2*hidden_dim_base, hidden_dim_base),
            nn.Linear(hidden_dim_base, 1)
            )

    def forward(self, x, y, hidden):
        x = self.sample_block(x)
        y = self.label_block(y)
        out = torch.cat([x, y], dim=2)
        out, _ = self.lstm(out, hidden)
        validity = self.output_block(out)
        return validity

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers, batch_size, 2*self.hidden_dim_base).double(),
                      torch.zeros(self.n_layers, batch_size, 2*self.hidden_dim_base).double())
        return hidden
