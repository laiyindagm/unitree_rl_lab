# Encoder：吃 H × D_obs 的历史，输出 (vel_3 + latent_16)
enc_input_dim = self.temporal_steps * self.num_one_step_obs
enc_layers = []
for l in range(len(enc_hidden_dims) - 1):
    enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
    enc_input_dim = enc_hidden_dims[l]
enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
self.encoder = nn.Sequential(*enc_layers)

# Target：吃 1 × D_obs 的下一帧，输出 latent_16
tar_input_dim = self.num_one_step_obs
tar_layers = []
for l in range(len(tar_hidden_dims)):
    tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation]
    tar_input_dim = tar_hidden_dims[l]
tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
self.target = nn.Sequential(*tar_layers)