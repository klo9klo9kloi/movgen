import torch
import torch.nn as nn

class PS_G(nn.Module):
    def __init__(self, pose_dim, hidden_dim_base, n_layers=1):
        super(PS_G, self).__init__()

        self.input_block = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim_base),
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
        self.end_pose_block = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim_base),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.lstm = nn.LSTM(2*hidden_dim_base, 2*hidden_dim_base, n_layers)
        self.output_block = nn.Sequential(
            nn.Linear(2*hidden_dim_base, pose_dim),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.pose_dim = pose_dim
        self.n_layers = n_layers
        self.hidden_dim_base = hidden_dim_base

    def forward(self, p_t, p_T, hidden):
        # y shape (T, B, D)
        p_t = self.input_block(p_t)
        p_T = self.end_pose_block(p_T)

        p_concat = torch.cat([p_t, p_T], dim=2)
        out, _ = self.lstm(p_concat, hidden) # shape (T, B, 2H)
        p_t_to_T = self.output_block(out)
        return p_t_to_T

    def init_hidden(self, batch_size, device):
        hidden = (torch.zeros(self.n_layers, batch_size, 2*self.hidden_dim_base).to(device).double(),
                      torch.zeros(self.n_layers, batch_size, 2*self.hidden_dim_base).to(device).double())
        return hidden

# VALIDITY PER POSE
class PS_D(nn.Module):
    def __init__(self, pose_dim, hidden_dim_base, n_layers=1):
        super(PS_D, self).__init__()

        self.sample_block = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim_base),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.end_pose_block = nn.Sequential( 
            nn.Linear(pose_dim, hidden_dim_base),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.lstm = nn.LSTM(2*hidden_dim_base, 2*hidden_dim_base, n_layers)
        self.output_block = nn.Sequential(
            nn.Linear(2*hidden_dim_base, hidden_dim_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim_base, 1) # decide validity of each point
        )
        self.n_layers = n_layers
        self.hidden_dim_base = hidden_dim_base

    def forward(self, sample, end_pose, hidden):
        sample = self.sample_block(sample)
        end_pose = self.end_pose_block(end_pose)
        seq_concat = torch.cat([sample, end_pose], dim=2)
        out, _ = self.lstm(seq_concat, hidden)
        validity = self.output_block(out)
        return validity

    def init_hidden(self, batch_size, device):
        hidden = (torch.zeros(self.n_layers, batch_size, 2*self.hidden_dim_base).to(device).double(),
                      torch.zeros(self.n_layers, batch_size, 2*self.hidden_dim_base).to(device).double())
        return hidden

# VALIDITY PER SEQUENCE
# class PS_D(nn.Module):
#     def __init__(self, pose_dim, hidden_dim_base, seq_len=15):
#         super(PS_D, self).__init__()

#         self.sample_block = nn.Sequential(
#             nn.Linear(pose_dim * seq_len, hidden_dim_base),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.end_pose_block = nn.Sequential( 
#             nn.Linear(pose_dim, hidden_dim_base//seq_len),
#             nn.LeakyReLU(0.2, inplace=True)
#         )

#         self.output_block = nn.Sequential(
#             nn.Linear(hidden_dim_base + hidden_dim_base//seq_len, hidden_dim_base*3 // 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(hidden_dim_base * 3 // 2, 1) # decide validity of each point
#         )
#         self.hidden_dim_base = hidden_dim_base

#     def forward(self, sample_seq, end_pose):
#         batch_size = sample_seq.size(1)
#         sample_seq = sample_seq.transpose(1, 0).reshape(batch_size, -1)
#         sample_seq = self.sample_block(sample_seq)
#         end_pose = self.end_pose_block(end_pose)
#         conditioned = torch.cat([sample_seq, end_pose], dim=1)
#         validity = self.output_block(conditioned)
#         return validity