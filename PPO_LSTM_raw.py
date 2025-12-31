import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

"""
Instead of using LSTM module, will be implementing LSTM from scratch.

Optionally could use GRU.

"""

class LSTM_cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_cell, self).__init__()
        #all needed inputs
        self.input_size = input_size
        self.hidden_size = hidden_size
        combined_size = self.input_size + self.hidden_size
        self.forget_gate = nn.Linear(combined_size, hidden_size)
        self.input_gate = nn.Linear(combined_size, hidden_size)
        self.candidate_gate = nn.Linear(combined_size, hidden_size)
        self.output_gate = nn.Linear(combined_size, hidden_size)

    def forward(self, i, hidden_state):
        h_prev, c_prev = hidden_state

        combined = torch.cat([i, h_prev], dim=1)

        if len(combined) != combined_size:
            #throw error combined size does not match the catted input_size + hidden_size
            return

        #forget gate uses binary classification (sigmoid)
        f_g = torch.sigmoid(self.forget_gate(combined))

        #input gate uses binary classification (sigmoid)
        i_g = torch.sigmoid(self.input_gate(combined))

        #candidate gate uses tanh, allows for positive and negative updates
        c_g = torch.tanh(self.candidate_gate(combined))

        #output gate uses binary classification (sigmoid)
        o_g = torch.sigmoid(self.output_gate(combined))

        #Follow LSTM rules, tensor algebra
        c_new = f_g * c_prev + i_g * c_g

        h_new = o_g * torch.tanh(c_new)

        return h_new, c_new

#process sequences
class LSTM_Layer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_Layer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_cell = LSTM_cell(input_size, hidden_size)
    
    """
    x: (batch_size, seq_len, input_size)
    hidden_state: tuple of (h0, c0) or None
    """
    def forward(self, x, hidden_state=None):
        batch_size, seq_len, input_size = X.shape
        if hidden_state == None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h , c = hidden_state

        #process embeddings
        embeddings = []

        #Process timesteps to get the embeddings (these are just stacks of hidden states, ignore cell states)
        for t in range(0, seq_len):
            #input here is
            h, c = self.lstm_cell(x[:, t, :], (h,c))
            embeddings.append(h)


        embeddings = torch.stack(embeddings, dim=1)

        return embeddings, (h, c)

        

class LSTM_PPO(nn.Module):
    def __init__(self, obs_dim, action_dim, lstm_hidden_size, dense_layers: list[128, 128], continous_actions):
        super(LSTM_PPO, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.dense_layers = dense_layers
        self.continous_actions = continous_actions
        self.lstm = LSTM_Layer(obs_dim, lstm_hidden_size)

        #Following Code will use FCNET to compute embeddings
        #Embeddings will be feed through this

        layers = []
        #outsize defaulted as 128
        in_size = lstm_hidden_size
        #size of 2
        for out_size in dense_layers:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.RELU())
            in_size = out_size
        # REUL(RELU(lstm_hidden_size x out_size),  out_size) -> outsize
        self._embeddings_fc_net = nn.Sequential(*layers)
        #Actor is the Policy head P(s,a)
        self.actor = nn.Linear(lstm_hidden_size, self.action_dim)
        #critic only ouputs a Value(s)
        self.critic = nn.Linear(lstm_hidden_size, 1)


    def _compute_embeddings_and_state_out(self, batch):
        obs = batch['obs']
        state_in = batch.get("state_in", None)

        if state_in is not None:
            h = state_in['h']
            c = state_out['c']
            hidden_state = (h,c)
        else:
            hidden_state = None
        
        embeddings, (h_new, c_new) = self.lstm(obs, hidden_state)

        embeddings_out = self._embeddings_fc_net(embeddings)

        return embeddings_out, {"h": h_new, "c": c_new}


    def forward(self, batch):

        embeddings, state_outs = self._compute_embeddings_and_state_out(batch)

        action_logits = self.actor(embeddings)

        return action_logits

    def compute_values(self, batch, embeddings):
        if embeddings is None:
            embeddings, state_outs = self._compute_embeddings_and_state_out(batch)
        values = self.critic(embeddings).squeeze(-1)
        return values

        















