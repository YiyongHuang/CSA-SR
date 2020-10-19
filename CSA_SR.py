import torch
from torch import nn
from data_process import *
# from ConvGRU import SeqVLADModule
from ConvGRU_att import SeqVLADModule
from SLSTM import SLSTM
import math
from torch import Tensor
from torch.nn import Parameter

word_counts, unk_required = build_vocab(word_count_threshold=0)
word2id, id2word = word_to_ids(word_counts, unk_requried=unk_required)


class CSA_SR(nn.Module):
    def __init__(self, vocab_size, batch_size=10, hidden=512, dropout=0.5, n_step=80, feats_c=1536,
                 feats_h=8, feats_w=8, num_centers=32, redu_dim=512):
        super(CSA_SR, self).__init__()
        self.batch_size = batch_size
        self.hidden = hidden
        self.n_step = n_step
        self.feats_c = feats_c
        self.feats_h = feats_h
        self.feats_w = feats_w
        self.num_centers = num_centers
        self.redu_dim = redu_dim

        # semantic weights
        self.w_s = Parameter(Tensor(300, self.hidden))

        self.w_x = Parameter(Tensor(self.hidden, self.hidden))

        self.u_x = Parameter(Tensor(self.hidden, self.hidden))

        self.reset_weigths()

        self.seqvlad = SeqVLADModule(self.n_step, self.num_centers, self.redu_dim)

        self.drop = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(self.num_centers*self.redu_dim, self.hidden)
#         self.linear1 = nn.Linear(self.num_centers, 1)
        self.linear2 = nn.Linear(2*self.hidden+self.redu_dim, vocab_size)

        # self.lstm1 = nn.LSTM(hidden, hidden, batch_first=True, dropout=dropout)
        self.lstm2 = SLSTM(2 * hidden, hidden)
        self.sem_decoder = nn.LSTM(hidden, 300, batch_first=True)
#         self.lstm2 = nn.LSTM(2*hidden, hidden, batch_first=True, dropout=dropout)

        self.embedding = nn.Embedding(vocab_size, hidden)

    def reset_weigths(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(self.redu_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
            
    def mean_pool_hiddens(hiddens, caption_masks):
        caption_lens = caption_masks.sum(dim=0).type(torch.cuda.FloatTensor)
        caption_masks = caption_masks.unsqueeze(2).expand_as(hiddens).type_as(hiddens)
        hiddens_masked = caption_masks * hiddens
        hiddens_mean_pooled = hiddens_masked.sum(dim=0) / \
            caption_lens.unsqueeze(1).expand(caption_lens.size(0), hiddens_masked.size(2))
        return hiddens_mean_pooled

    def forward(self, video, tag, caption=None):
        video = video.contiguous().view(-1, self.feats_c, self.feats_h, self.feats_w)
        if self.training:
            video = self.drop(video)

        vlad = self.seqvlad(video)  # batch_size, num_centers*redu_dim
        vlad = self.linear1(vlad)
        vlad = (vlad @ self.w_x) * (tag @ self.w_s) @ self.u_x
        vid_out = vlad.unsqueeze(1).repeat(1, self.n_step - 1, 1)

        if self.training:
            caption = self.embedding(caption[:, 0:self.n_step - 1])
            caption = torch.cat((caption, vid_out), 2)  # caption input

#             cap_out, state_cap = self.lstm2(caption)
            cap_out, state_cap = self.lstm2(caption, tag)
    
            sem_out, sem_state = self.sem_decoder(cap_out)
            sem_out = sem_out.sum(1)/(self.n_step - 1)

            cap_out = torch.cat((cap_out, caption), 2)
            cap_out = cap_out.contiguous().view(-1, 2*self.hidden+self.redu_dim)
            cap_out = self.drop(cap_out)
            cap_out = self.linear2(cap_out)
            return cap_out, sem_out
            # cap_out size [batch_size*79, vocab_size]
        else:
            bos_id = word2id['<BOS>'] * torch.ones(self.batch_size, dtype=torch.long)
            bos_id = bos_id.cuda()
            cap_input = self.embedding(bos_id)
            cap_input = torch.cat((cap_input, vid_out[:, 0, :]), 1)
            cap_input = cap_input.view(self.batch_size, 1, 2 * self.hidden)

#             cap_out, state_cap = self.lstm2(cap_input)
            cap_out, state_cap = self.lstm2(cap_input, tag)
            cap_out = torch.cat((cap_out, cap_input), 2)
            cap_out = cap_out.contiguous().view(-1, 2*self.hidden+self.redu_dim)
            cap_out = self.linear2(cap_out)
            cap_out = torch.argmax(cap_out, 1)
            # input ["<BOS>"] to let the generate start

            caption = []
            caption.append(cap_out)
            # put the generate word index in caption list, generate one word at one time step for each batch
            for i in range(self.n_step - 2):
                cap_input = self.embedding(cap_out)
                cap_input = torch.cat((cap_input, vid_out[:, 1 + i, :]), 1)
                cap_input = cap_input.view(self.batch_size, 1, 2 * self.hidden)

#                 cap_out, state_cap = self.lstm2(cap_input, state_cap)
                cap_out, state_cap = self.lstm2(cap_input, tag, state_cap)
                cap_out = torch.cat((cap_out, cap_input), 2)
                cap_out = cap_out.contiguous().view(-1, 2*self.hidden+self.redu_dim)
                cap_out = self.linear2(cap_out)
                cap_out = torch.argmax(cap_out, 1)
                # get the index of each word in vocabulary
                caption.append(cap_out)
            return caption
            # size of caption is [79, batch_size]
