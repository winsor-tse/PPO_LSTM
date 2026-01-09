import torch
import torch.nn as nn

class LSTM_cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_cell, self).__init__()
        #all needed inputs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.combined_size = self.input_size + self.hidden_size
        self.forget_gate = nn.Linear(self.combined_size, hidden_size)
        self.input_gate = nn.Linear(self.combined_size, hidden_size)
        self.candidate_gate = nn.Linear(self.combined_size, hidden_size)
        self.output_gate = nn.Linear(self.combined_size, hidden_size)

    def forward(self, i, hidden_state):
        h_prev, c_prev = hidden_state

        combined = torch.cat([i, h_prev], dim=1)

        if combined.shape[1] != self.combined_size:
            raise ValueError("combined input dimension does not match input_size + hidden_size")

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
        batch_size, seq_len, input_size = x.shape
        if hidden_state is None:
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