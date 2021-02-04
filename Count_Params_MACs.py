import torch
from torch import nn
from thop import profile
import numpy as np
from SLSTM import *
from SGRU_RCN_reconstruct_sem import *

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

seqvlad = SeqVLADModule(40, 32, 512)
batch_size = 64
channel_size = 40*32
# ca = ChannelAttention(channel_size)
# se = SLayer()
# lstm2 = SLSTM(1024, 512)
# lstm = nn.LSTM(512, 300, batch_first=True)
# s2vt = S2VT(vocab_size=13010, batch_size=batch_size, n_step=40)
inp = torch.randn(40, 1536, 8, 8)
# inp = torch.randn(64, channel_size, 8, 8)
MACs, params = profile(seqvlad, inputs=(inp,))
print(MACs)

print_network(seqvlad)