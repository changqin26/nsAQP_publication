import torch
import torch.nn as nn
from typing import Tuple  # Import Tuple for type annotations

class TreeLSTM(nn.Module):
    def __init__(self, hidden_dim: int):
        super(TreeLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # Weight matrices for input and concatenated hidden states
        self.W_i = nn.Linear(hidden_dim, hidden_dim)
        self.U_i = nn.Linear(2 * hidden_dim, hidden_dim)

        self.W_f = nn.Linear(hidden_dim, hidden_dim)
        self.U_f_left = nn.Linear(2 * hidden_dim, hidden_dim)
        self.U_f_right = nn.Linear(2 * hidden_dim, hidden_dim)

        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.U_o = nn.Linear(2 * hidden_dim, hidden_dim)

        self.W_u = nn.Linear(hidden_dim, hidden_dim)
        self.U_u = nn.Linear(2 * hidden_dim, hidden_dim)

        # Xavier initialization for the weights
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        left: Tuple[torch.Tensor, torch.Tensor],
        right: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Instead of using tuple unpacking, access elements via indexing:
        h_left = left[0]
        c_left = left[1]
        h_right = right[0]
        c_right = right[1]

        # Concatenate hidden states of left and right children => shape [batch_size, 2*hidden_dim]
        h_cat = torch.cat([h_left, h_right], dim=1)

        # Forget gate (right)
        pre_f_right = self.W_f(x) + self.U_f_right(h_cat)
        f_right = torch.sigmoid(pre_f_right) + 1e-3

        # Input gate
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_cat))

        # Forget gate (left)
        pre_f_left = self.W_f(x) + self.U_f_left(h_cat)
        f_left = torch.sigmoid(pre_f_left)

        # Output gate
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_cat))

        # Candidate cell state
        u = torch.tanh(self.W_u(x) + self.U_u(h_cat))

        # Update cell state
        c = i * u + f_left * c_left + f_right * c_right

        # Compute hidden state
        h = o * torch.tanh(c)

        return h, c
