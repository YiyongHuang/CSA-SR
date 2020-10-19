import torch
from torch import nn
import math
from torch import Tensor
from torch.nn import Parameter


class SLSTM(nn.Module):
    """semantic LSTM like nn.LSTM"""

    def __init__(self, input_size: int, hidden_size: int):
        super(SLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ii = Parameter(Tensor(hidden_size, hidden_size))
        self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(Tensor(hidden_size, 1))
        self.b_hi = Parameter(Tensor(hidden_size, 1))

        self.w_if = Parameter(Tensor(hidden_size, hidden_size))
        self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(Tensor(hidden_size, 1))
        self.b_hf = Parameter(Tensor(hidden_size, 1))

        self.w_io = Parameter(Tensor(hidden_size, hidden_size))
        self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(Tensor(hidden_size, 1))
        self.b_ho = Parameter(Tensor(hidden_size, 1))

        self.w_ig = Parameter(Tensor(hidden_size, hidden_size))
        self.w_hg = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(Tensor(hidden_size, 1))
        self.b_hg = Parameter(Tensor(hidden_size, 1))

        # semantic weights
        self.w_is = Parameter(Tensor(hidden_size, 300))
        self.w_fs = Parameter(Tensor(hidden_size, 300))
        self.w_gs = Parameter(Tensor(hidden_size, 300))
        self.w_os = Parameter(Tensor(hidden_size, 300))

        self.u_is = Parameter(Tensor(hidden_size, 300))
        self.u_fs = Parameter(Tensor(hidden_size, 300))
        self.u_gs = Parameter(Tensor(hidden_size, 300))
        self.u_os = Parameter(Tensor(hidden_size, 300))

        self.w_ix = Parameter(Tensor(hidden_size, input_size))
        self.w_fx = Parameter(Tensor(hidden_size, input_size))
        self.w_gx = Parameter(Tensor(hidden_size, input_size))
        self.w_ox = Parameter(Tensor(hidden_size, input_size))

        self.u_ih = Parameter(Tensor(hidden_size, hidden_size))
        self.u_fh = Parameter(Tensor(hidden_size, hidden_size))
        self.u_gh = Parameter(Tensor(hidden_size, hidden_size))
        self.u_oh = Parameter(Tensor(hidden_size, hidden_size))

        self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs, tag, state=None):
        """Forward
        Args:
            inputs: [batch_size, seq_size, input_size]
            state: ([hidden_size, batch_size], [hidden_size, batch_size])
        """

        batch_size, seq_size, _ = inputs.size()
        tag = tag.t()
        
        if state is None:
            if torch.cuda.is_available():
                h_t = torch.zeros(self.hidden_size, batch_size).cuda()
                c_t = torch.zeros(self.hidden_size, batch_size).cuda()
            else:
                h_t = torch.zeros(self.hidden_size, batch_size)
                c_t = torch.zeros(self.hidden_size, batch_size)
        else:
            (h_t, c_t) = state

        hidden_seq = []

#         seq_size = 1
        for t in range(seq_size):
            x = inputs[:, t, :].t()
            # print(x.shape)
            # input gate

            xi = (self.w_is @ tag) * (self.w_ix @ x)
            xf = (self.w_fs @ tag) * (self.w_fx @ x)
            xg = (self.w_gs @ tag) * (self.w_gx @ x)
            xo = (self.w_os @ tag) * (self.w_ox @ x)

            hi = (self.u_is @ tag) * (self.u_ih @ h_t)
            hf = (self.u_fs @ tag) * (self.u_fh @ h_t)
            hg = (self.u_gs @ tag) * (self.u_gh @ h_t)
            ho = (self.u_os @ tag) * (self.u_oh @ h_t)

            i = torch.sigmoid(self.w_ii @ xi + self.b_ii + self.w_hi @ hi +
                              self.b_hi)
            # forget gate
            f = torch.sigmoid(self.w_if @ xf + self.b_if + self.w_hf @ hf +
                              self.b_hf)
            # cell
            g = torch.tanh(self.w_ig @ xg + self.b_ig + self.w_hg @ hg
                           + self.b_hg)
            # output gate
            o = torch.sigmoid(self.w_io @ xo + self.b_io + self.w_ho @ ho +
                              self.b_ho)

            # hidden, batch
            c_t = f * c_t + i * g
            h_t = o * torch.tanh(c_t)
            # print(h_t.shape)
            # c_t = c_t.t().unsqueeze(0)
            h_out = h_t.t().unsqueeze(1)
            hidden_seq.append(h_out)

        hidden_seq = torch.cat(hidden_seq, dim=1)
        return hidden_seq, (h_t, c_t)

