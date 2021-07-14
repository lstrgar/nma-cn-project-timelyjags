import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lib.distributions.normal import Normal
from torchvision import models


class TrajectoryModel(nn.Module):
    """
    This is more or less a copy of the TREBA TVAE definition
    """

    def __init__(self, model_config, train_data):
        """
        config is a dictionary of model configurations
        train_data - used to set the input dimensions to the model
        """
        super().__init__()

        # Dimension definitions
        self.rnn_dim = model_config["rnn_dim"]
        self.num_layers = model_config["num_layers"]

        # input_dim should be the dimensionality of each element in the sequence
        self.input_dim = train_data[0].shape[-1]
        self.output_dim = train_data[0].shape[-1]

        # z_dim is the dimension of the latent variable
        self.z_dim = model_config["z_dim"]

        # h_dim is the dimension of the recurrent portion
        self.h_dim = model_config["h_dim"]

        # GRU part of encoder
        self.enc_birnn = nn.GRU(
            self.input_dim,
            hidden_size=self.h_dim,
            num_layers=self.num_layers,
            bidirectional=True,
        )

        # fc part of encoder
        self.enc_fc = nn.Sequential(
            nn.Linear(2 * self.rnn_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        # Mean and logvar
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)
        self.enc_logvar = nn.Linear(self.h_dim, self.z_dim)

        # Recurrent portion of the decoder
        self.dec_rnn = nn.GRU(
            self.output_dim * 2, self.rnn_dim, num_layers=self.num_layers
        )

        # Fully-connected portion of the decoder
        self.dec_action_fc = nn.Sequential(
            nn.Linear(self.output_dim + self.z_dim + self.rnn_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        # Parameters for the distribution from which we sample our desired outputs
        self.dec_action_mean = nn.Linear(self.h_dim, self.output_dim)
        self.dec_action_logvar = nn.Linear(self.h_dim, self.output_dim)

    def encode(self, states):
        """
        This returns a Normal distribution with the learned mean
        and variance
        """
        # enc_birnn_input = torch.cat([states, actions], dim=-1)
        enc_birnn_input = states
        hiddens, _ = self.enc_birnn(enc_birnn_input)

        # Average over time ???
        avg_hiddens = torch.mean(hiddens, dim=0)

        # Pass output of last hidden layer
        # avg_hiddens =  hiddens[-1]

        # Pass through FC portion of encoder
        enc_fc_input = avg_hiddens
        enc_h = self.enc_fc(enc_fc_input)

        # Get the mean and logvar
        enc_mean = self.enc_mean(enc_h)
        enc_logvar = self.enc_logvar(enc_h)

        return states, Normal(enc_mean, enc_logvar)

    def decode_action(self, state):
        dec_fc_input = torch.cat([state, self.z], dim=-1)
        dec_fc_input = torch.cat([dec_fc_input, self.hidden[-1]], dim=1)

        dec_h = self.dec_action_fc(dec_fc_input)
        dec_mean = self.dec_action_mean(dec_h)

        if isinstance(self.dec_action_logvar, nn.Parameter):
            dec_logvar = self.dec_action_logvar
        else:
            dec_logvar = self.dec_action_logvar(dec_h)

        return Normal(dec_mean, dec_logvar)

    def reset_policy(
        self, z, labels=None, temperature=0.01, num_samples=0, device="cpu"
    ):
        if z is None:
            assert num_samples > 0
            assert device is not None
            z = torch.randn(num_samples, self.config["z_dim"]).to(device)

        self.z = z
        self.temperature = temperature
        self.hidden = self.init_hidden_state(batch_size=z.size(0)).to(z.device)

    def init_hidden_state(self, batch_size=1):
        return torch.zeros(self.num_layers, batch_size, self.rnn_dim)

    def update_hidden(self, state, action):
        state_action_pair = torch.cat([state, action], dim=1).unsqueeze(0)
        hiddens, self.hidden = self.dec_rnn(state_action_pair, self.hidden)

        return hiddens

    def forward(self, in_features):
        """
        Perform a forward pass on the TVAE only
        """

        # Get orig extracted features, q_{phi}(z | x) --> the input features include the actions
        x, posterior = self.encode(in_features)

        # Gather the desired output states
        y = x
        out_states = x

        # Calculate the desired output actions
        out_actions = y[:, 1:, :] - y[:, :-1, :]

        # Transposed out states and actions
        out_states = out_states.transpose(0, 1)  #  [seq_len, batch_size, output_dim]
        out_actions = out_actions.transpose(0, 1)  #  [seq_len, batch_size, output_dim]

        # Take the kld between the posterior and Gaussian prior --> Good to go
        # kld = Normal.kl_divergence(posterior, free_bits=0.0).detach()

        # Set the free bits --> Good to go
        kld = Normal.kl_divergence(posterior, free_bits=1 / self.z_dim)
        kld = torch.sum(kld)

        # Initialize the hidden state of the decoder
        self.reset_policy(z=posterior.sample())

        seq_nll = 0

        # No teacher forcing
        dec_state = out_states[0]
        for t in range(out_actions.size(0)):
            # Get distribution of actions based on the state
            action_likelihood = self.decode_action(dec_state)

            # Calculate the likelihood of the true action under our learned distribution
            seq_nll -= action_likelihood.log_prob(out_actions[t])

            # Sample an action
            dec_action = action_likelihood.sample()

            # Construct a new state from the action
            dec_state = dec_state + dec_action

            # Update with synthetic state and action
            self.update_hidden(dec_state, dec_action)

        return seq_nll, kld

    def decode(self, z_in, out_states):

        self.reset_policy(z_in)

        seq_nll = 0

        # No teacher forcing
        dec_state = out_states[0]
        out_states = [dec_state]
        for t in range(out_actions.size(0)):
            # Get distribution of actions based on the state
            action_likelihood = self.decode_action(dec_state)

            # Calculate the likelihood of the true action under our learned distribution
            seq_nll -= action_likelihood.log_prob(out_actions[t])

            # Sample an action
            dec_action = action_likelihood.sample()

            # Construct a new state from the action
            dec_state = dec_state + dec_action

            # Append decoded state to output
            out_states.append(dec_state)

            # Update with synthetic state and action
            self.update_hidden(dec_state, dec_action)

        return torch.stack(out_states, dim=1).squeeze(), z

    def forward_test(self, in_features):
        """
        Perform a forward pass on the TVAE only
        """
        # Get orig extracted features, q_{phi}(z | x) --> the input features include the actions
        x, posterior = self.encode(in_features)

        # Gather the desired output states
        y = x
        out_states = x

        # Calculate the desired output actions
        out_actions = y[:, 1:, :] - y[:, :-1, :]

        # Transposed out states and actions
        out_states = out_states.transpose(0, 1)  #  [seq_len, batch_size, output_dim]
        out_actions = out_actions.transpose(0, 1)  #  [seq_len, batch_size, output_dim]

        # Take the kld between the posterior and Gaussian prior --> Good to go
        # kld = Normal.kl_divergence(posterior, free_bits=0.0).detach()

        # Set the free bits --> Good to go
        kld = Normal.kl_divergence(posterior, free_bits=1 / self.z_dim)
        kld = torch.sum(kld)

        # Initialize the hidden state of the decoder
        self.reset_policy(z=posterior.sample())

        seq_nll = 0

        # No teacher forcing
        dec_state = out_states[0]
        out_states = [dec_state]
        for t in range(out_actions.size(0)):
            # Get distribution of actions based on the state
            action_likelihood = self.decode_action(dec_state)

            # Calculate the likelihood of the true action under our learned distribution
            seq_nll -= action_likelihood.log_prob(out_actions[t])

            # Sample an action
            dec_action = action_likelihood.sample()

            # Construct a new state from the action
            dec_state = dec_state + dec_action

            # Append decoded state to output
            out_states.append(dec_state)

            # Update with synthetic state and action
            self.update_hidden(dec_state, dec_action)

        return torch.stack(out_states, dim=1).squeeze(), z


class GlobalHiddenTrajectoryModel(TrajectoryModel):
    def __init__(
        self,
        model_config,
        train_data,
        pretrained_feat_extractor=True,
        freeze_feat_extractor=True,
    ):

        feat_encoder = models.resnet34(pretrained=pretrained_feat_extractor)
        feat_encoder.fc = nn.Identity()

        with torch.no_grad():
            train_data = feat_encoder(train_data[0])

        # Initialize from base class
        super(GlobalHiddenTrajectoryModel, self).__init__(model_config, train_data)

        self.feat_encoder = feat_encoder

        # fc part of encoder
        self.enc_fc = nn.Sequential(
            nn.Linear(2 * model_config["seq_len"] * self.rnn_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        if not pretrained_feat_extractor and freeze_feat_extractor:
            raise ValueError("Cannot freeze feat extractor when not feat extractor")

        # not passing gradients to feat extractor
        # in future may want to fine tune later layers only
        if freeze_feat_extractor:
            for param in self.feat_encoder.parameters():
                param.requires_grad = False

    def encode(self, states):
        """
        This returns a Normal distribution with the learned mean 
        and variance
        """
        # Extract lower dim features
        tmp = torch.empty((states.size(0), states.size(1), 512))
        for i in range(tmp.size(0)):
            tmp[i] = self.feat_encoder(states[i])
        states = tmp

        enc_birnn_input = states  # torch.cat([states, actions], dim=-1)
        hiddens, _ = self.enc_birnn(enc_birnn_input)
        if torch.sum(torch.isnan(hiddens)) > 0:
            print("wut")

        # torch.empty messes upt the device ...
        tmp_device = hiddens.device

        # Concatenate the outputs at each time step for every batch
        enc_fc_input = torch.flatten(hiddens, 1)

        # Pass through FC portion of encoder
        enc_h = self.enc_fc(enc_fc_input)

        # Get the mean and logvar
        enc_mean = self.enc_mean(enc_h)
        enc_logvar = self.enc_logvar(enc_h)

        return states, Normal(enc_mean, enc_logvar)

