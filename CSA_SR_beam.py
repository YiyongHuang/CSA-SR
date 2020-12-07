import torch
from torch import nn
from data_process import *
from ConvGRU_att import SeqVLADModule
from SLSTM import SLSTM


word_counts, unk_required = build_vocab(word_count_threshold=0)
word2id, id2word = word_to_ids(word_counts, unk_requried=unk_required)


class CSA_SR(nn.Module):
    def __init__(self, vocab_size, batch_size=64, hidden=512, dropout=0.5, n_step=40, feats_c=1536,
                 feats_h=8, feats_w=8, num_centers=32, redu_dim=512, beam=5):
        super(CSA_SR, self).__init__()
        self.batch_size = batch_size
        self.hidden = hidden
        self.n_step = n_step
        self.feats_c = feats_c
        self.feats_h = feats_h
        self.feats_w = feats_w
        self.num_centers = num_centers
        self.redu_dim = redu_dim
        self.vocab_size = vocab_size
        self.beam = beam

        self.seqvlad = SeqVLADModule(self.n_step*2-1, self.num_centers, self.redu_dim)

        self.drop = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(self.num_centers, 1)
        self.linear2 = nn.Linear(hidden, vocab_size)

        # self.lstm1 = nn.LSTM(hidden, hidden, batch_first=True, dropout=dropout)
        self.lstm2 = SLSTM(2*hidden, hidden)

        self.embedding = nn.Embedding(vocab_size, hidden)

    def forward(self, video, tag, caption=None):
        padding = torch.zeros([self.batch_size, self.n_step - 1, self.feats_c, self.feats_w, self.feats_h]).cuda()
        video = torch.cat((video, padding), 1)
        video = video.contiguous().view(-1, self.feats_c, self.feats_h, self.feats_w)
        video = self.drop(video)
        # video = self.linear1(video)                   # video embed

        vlad = self.seqvlad(video)                # batch_size, timesteps, num_centers, redu_dim
        vlad = vlad.transpose(3, 2)
        vlad = self.linear1(vlad)
        vid_out = vlad.squeeze(3)                  # batch_size, timesteps, redu_dim
        # video = video.view(-1, self.n_step, self.hidden)
        # padding = torch.zeros([self.batch_size, self.n_step-1, self.hidden]).cuda()
        # video = torch.cat((vlad, padding), 1)        # video input
        # vid_out, state_vid = self.lstm1(video)

        if self.training:
            caption = self.embedding(caption[:, 0:self.n_step-1])
            padding = torch.zeros([self.batch_size, self.n_step, self.hidden]).cuda()
            caption = torch.cat((padding, caption), 1)        # caption padding
            caption = torch.cat((caption, vid_out), 2)        # caption input

            cap_out, state_cap = self.lstm2(caption, tag)
            # size of cap_out is [batch_size, 2*n_step-1, hidden]
            cap_out = cap_out[:, self.n_step:, :]
            cap_out = cap_out.contiguous().view(-1, self.hidden)
            cap_out = self.drop(cap_out)
            cap_out = self.linear2(cap_out)
            return cap_out
            # cap_out size [batch_size*79, vocab_size]
        else:
            padding = torch.zeros([self.batch_size, self.n_step, self.hidden]).cuda()
            cap_input = torch.cat((padding, vid_out[:, 0:self.n_step, :]), 2)
            cap_out, state_cap = self.lstm2(cap_input, tag)
            # padding input of the second layer of LSTM, 80 time steps

            bos_id = word2id['<BOS>']*torch.ones(self.batch_size, dtype=torch.long)
            bos_id = bos_id.cuda()
            cap_input = self.embedding(bos_id)
            cap_input = torch.cat((cap_input, vid_out[:, self.n_step, :]), 1)
            cap_input = cap_input.view(self.batch_size, 1, 2*self.hidden)

            cap_out, state_cap = self.lstm2(cap_input, tag, state_cap)
            cap_out = cap_out.contiguous().view(-1, self.hidden)
            cap_out = self.drop(cap_out)
            cap_out = self.linear2(cap_out)
#             cap_out = torch.argmax(cap_out, 1)
            odd, cap_out = cap_out.topk(self.beam, 1, True, True)    # [batch, self.beam]
            # input ["<BOS>"] to let the generate start
            cap_out = cap_out.cpu().numpy()    # [self.beam, 1]
            result = cap_out.copy()
            result = result.transpose(1, 0).tolist()
#             print(result)
            
            hid_state = []
            cap_idx = [np.zeros(self.beam, dtype=int)]
            for i in range(self.beam):
                hid_state.append(state_cap)
                
            cap_odds = []
#             caption = []
#             caption.append(cap_out)
            # put the generate word index in caption list, generate one word at one time step for each batch
            for i in range(self.n_step-62):
                cap_out = torch.from_numpy(cap_out).cuda()
                for j in range(self.beam):
                    cap_input = cap_out[:, j]
                    cap_input = self.embedding(cap_input)
                    cap_input = torch.cat((cap_input, vid_out[:, self.n_step+1+i, :]), 1)
                    cap_input = cap_input.view(self.batch_size, 1, 2 * self.hidden)

                    cap_step, state_cap = self.lstm2(cap_input, tag, hid_state[cap_idx[0][j]])

                    cap_step = cap_step.contiguous().view(-1, self.hidden)
                    cap_step = self.drop(cap_step)
                    cap_step = self.linear2(cap_step)
                    cap_step = odd[:, j].unsqueeze(1)*cap_step
                    cap_odds.append(cap_step)
                    hid_state.append(state_cap)
                    # cap_step = torch.argmax(cap_step, 1)
                # get the index of each word in vocabulary
                cap_odds = torch.cat(cap_odds, dim=1)
                odd, cap_out = cap_odds.topk(self.beam, 1, True, False)  # [batch, self.beam]
                # caption.append(cap_step)
                cap_out = cap_out.cpu().numpy()
                cap_idx = cap_out // self.vocab_size
#                 print(cap_idx)
                cap_out = cap_out % self.vocab_size    # [1, self.beam]
#                 cap_out = np.squeeze(cap_out)
#                 print(cap_out)
                hid_state = hid_state[self.beam:]
                cap_odds = []
                new_result = []
                for k in range(self.beam):
                    new_result.append(result[cap_idx[0][k]].copy())
#                     print(new_result)
#                     print(cap_out[0][k])
                    new_result[k].append(cap_out[0][k])
#                     print(new_result)
                result = new_result
#                 print(result)
#                 if i==3:
#                     break
                
#                 cap_input = self.embedding(cap_out)
#                 cap_input = torch.cat((cap_input, vid_out[:, self.n_step+1+i, :]), 1)
#                 cap_input = cap_input.view(self.batch_size, 1, 2 * self.hidden)

#                 cap_out, state_cap = self.lstm2(cap_input, tag, state_cap)
#                 cap_out = cap_out.contiguous().view(-1, self.hidden)
#                 cap_out = self.drop(cap_out)
#                 cap_out = self.linear2(cap_out)
#                 cap_out = torch.argmax(cap_out, 1)
#                 # get the index of each word in vocabulary
#                 caption.append(cap_out)
            num = torch.argmax(odd)
            caption = torch.from_numpy(np.array(result[num])).unsqueeze(1)
            return caption
            # size of caption is [79, batch_size]







