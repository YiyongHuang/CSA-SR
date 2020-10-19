import torch
import math
from torch import nn
from torch.autograd import Variable


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class Identity(torch.nn.Module):
    def forward(self, input):
        return input

    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.tanh(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) 



class SeqVLADModule(torch.nn.Module):

    def __init__(self, timesteps, num_centers, redu_dim, with_relu=False, activation=None, with_center_loss=False, init_method='xavier_normal'):
        '''
            num_centers: set the number of centers for sevlad
            redu_dim: reduce channels for input tensor
        '''
        super(SeqVLADModule, self).__init__()
        self.num_centers = num_centers
        self.redu_dim = redu_dim
        self.timesteps = timesteps
        self.with_relu = with_relu

        self.in_shape = None
        self.out_shape = self.num_centers * self.redu_dim
        self.batch_size = None
        self.activation = activation

        self.with_center_loss = with_center_loss

        self.init_method = init_method
        
        self.relu = nn.ReLU(inplace=True)
        if self.with_relu:
            print('redu with relu ...')

        def init_func(t):
            if self.init_method == 'xavier_normal':
                return torch.nn.init.xavier_normal(t)
            elif self.init_method == 'orthogonal':
                return torch.nn.init.orthogonal(t)
            elif self.init_method == 'uniform':
                return torch.nn.init.uniform(t, a=0, b=0.01)

        self.U_r = torch.Tensor(self.num_centers, self.num_centers, 3, 3)  # weight : out, in , h, w
        self.U_r = init_func(self.U_r)
        self.U_r = torch.nn.Parameter(self.U_r, requires_grad=True)

        self.U_z = torch.Tensor(self.num_centers, self.num_centers, 3, 3)  # weight : out, in , h, w
        self.U_z = init_func(self.U_z)
        self.U_z = torch.nn.Parameter(self.U_z, requires_grad=True)

        self.U_h = torch.Tensor(self.num_centers, self.num_centers, 3, 3)  # weight : out, in , h, w
        self.U_h = init_func(self.U_h)
        self.U_h = torch.nn.Parameter(self.U_h, requires_grad=True)

        self.w_x = torch.Tensor(self.num_centers, self.redu_dim, 1, 1)  # weight : out, in , h, w
        self.w_x = torch.nn.init.xavier_normal(self.w_x)
        self.w_x = torch.nn.Parameter(self.w_x, requires_grad=True)

        if self.redu_dim < 1024:
            self.redu_w = torch.Tensor(self.redu_dim, 1536, 1, 1)  # weight : out, in , h, w
            self.redu_w = torch.nn.init.xavier_normal(self.redu_w)
            self.redu_w = torch.nn.Parameter(self.redu_w, requires_grad=True)

            self.redu_b = torch.Tensor(self.redu_dim, )  # weight : out, in , h, w
            self.redu_b = torch.nn.init.uniform(self.redu_b)
            self.redu_b = torch.nn.Parameter(self.redu_b, requires_grad=True)

        self.centers = torch.Tensor(self.num_centers, self.redu_dim)  # weight : out, in , h, w
        self.centers = init_func(self.centers)
        self.centers = torch.nn.Parameter(self.centers, requires_grad=True)

        # self.share_b = torch.Tensor(self.num_centers, )  # weight : out, in , h, w
        # self.share_b = torch.nn.init.uniform(self.share_b)
        # self.share_b = torch.nn.Parameter(self.share_b, requires_grad=True)
        self.share_b = torch.Tensor(self.num_centers, )  # weight : out, in , h, w
        self.share_b = torch.nn.init.uniform(self.share_b)
        self.share_b = torch.nn.Parameter(self.share_b, requires_grad=True)

        self.share_w = torch.Tensor(self.num_centers, self.num_centers, 3, 3)  # weight : out, in , h, w
        self.share_w = init_func(self.share_w)
        self.share_w = torch.nn.Parameter(self.share_w, requires_grad=True)

        # attention params
#         self.senet = SELayer(self.timesteps*self.num_centers)
        self.ca = ChannelAttention(self.timesteps*self.num_centers)
#         self.sa = SpatialAttention()

        self.att_b = torch.Tensor(self.num_centers, )  # weight : out, in , h, w
        self.att_b = torch.nn.init.uniform(self.att_b)
        self.att_b = torch.nn.Parameter(self.att_b, requires_grad=True)

        self.att_h = torch.Tensor(self.num_centers, self.num_centers, 3, 3)  # weight : out, in , h, w
        self.att_h = init_func(self.att_h)
        self.att_h = torch.nn.Parameter(self.att_h, requires_grad=True)

        self.att_x = torch.Tensor(self.num_centers, self.num_centers, 3, 3)  # weight : out, in , h, w
        self.att_x = init_func(self.att_x)
        self.att_x = torch.nn.Parameter(self.att_x, requires_grad=True)

    def forward(self, input):
        '''
        input_tensor: N*timesteps, C, H, W
        '''
        self.in_shape = input.size()
        self.batch_size = self.in_shape[0] // self.timesteps
        if self.batch_size == 0:
            self.batch_size = 1

        input_tensor = input

        if self.redu_dim == None:
            self.redu_dim = self.in_shape[1]
        elif self.redu_dim < self.in_shape[1]:
            input_tensor = torch.nn.functional.conv2d(input_tensor, self.redu_w, bias=self.redu_b, stride=1, padding=0,
                                                      dilation=1, groups=1)
        if self.with_relu:
            input_tensor = torch.nn.functional.relu(input_tensor)

        self.out_shape = self.num_centers * self.redu_dim

        # wx_plus_b : N*timesteps, redu_dim, H, W
        wx_plus_b = torch.nn.functional.conv2d(input_tensor, self.w_x, bias=None, stride=1, padding=0,
                                               dilation=1, groups=1)

        att_v = torch.nn.functional.conv2d(wx_plus_b, self.att_x, bias=None, stride=1, padding=1)
        att_v = att_v.view(self.batch_size, self.timesteps, self.num_centers, self.in_shape[2],
                           self.in_shape[3]).transpose(1, 0)

        wx_plus_b = wx_plus_b.view(self.batch_size, self.timesteps, self.num_centers, self.in_shape[2],
                                   self.in_shape[3])
        ## reshape

        ## init hidden states
        ## h_tm1 = N, num_centers, H, W
        h_tm1 = torch.autograd.Variable(
            torch.Tensor(self.batch_size, self.num_centers, self.in_shape[2], self.in_shape[3]), requires_grad=True)
        h_tm1 = torch.nn.init.constant(h_tm1, 0).cuda()

        ## prepare the input tensor shape
        ## output
        assignments = []
        # input_tensor = input_tensor.view()
        for i in range(self.timesteps):
            att_h = torch.nn.functional.conv2d(h_tm1, self.att_h, bias=self.att_b, stride=1, padding=1)
            e = (att_v + att_h).transpose(1, 0).view(self.batch_size, self.timesteps*self.num_centers,
                                                                    self.in_shape[2], self.in_shape[3])
            e = self.relu(e)
            e = self.ca(e)
#             e = self.sa(e) * e
            e = e.squeeze().view(self.batch_size, self.timesteps, self.num_centers)
            e = e.transpose(1, 0).contiguous().view(self.timesteps, self.batch_size*self.num_centers)
            e = torch.exp(e)
            denominator = e.sum(dim=0)  # b*c
            denominator = denominator + denominator.eq(0).float()
            alphas = torch.div(e, denominator)  # n, b*c
            alphas = alphas.view(self.timesteps, self.batch_size, self.num_centers).transpose(1, 0).unsqueeze(3).\
                unsqueeze(4).repeat(1, 1, 1, self.in_shape[2], self.in_shape[3])
            alphas = alphas * wx_plus_b
            wx_plus_b_at_t = alphas.sum(dim=1)
            wx_plus_b_at_t = torch.nn.functional.conv2d(wx_plus_b_at_t, self.share_w, bias=self.share_b, stride=1, padding=1)
            # wx_plus_b_at_t = wx_plus_b[:, i, :, :, :]

            Uz_h = torch.nn.functional.conv2d(h_tm1, self.U_z, bias=None, stride=1, padding=1)
            z = torch.nn.functional.sigmoid(wx_plus_b_at_t + Uz_h)

            Ur_h = torch.nn.functional.conv2d(h_tm1, self.U_r, bias=None, stride=1, padding=1)
            r = torch.nn.functional.sigmoid(wx_plus_b_at_t + Ur_h)

            Uh_h = torch.nn.functional.conv2d(r * h_tm1, self.U_h, bias=None, stride=1, padding=1)
            hh = torch.nn.functional.tanh(wx_plus_b_at_t + Uh_h)

            h = (1 - z) * hh + z * h_tm1
            assignments.append(h)
            h_tm1 = h

        ## timesteps, batch_size , num_centers, h, w

        assignments = torch.stack(assignments, dim=0)

        ## timesteps, batch_size, num_centers, h, w ==> batch_size, timesteps, num_centers, h, w
        assignments = torch.transpose(assignments, 0, 1).contiguous()

        ## assignments: batch_size, timesteps, num_centers, h*w
        assignments = assignments.view(self.batch_size*self.timesteps, self.num_centers, self.in_shape[2]*self.in_shape[3])
        if self.activation is not None:
            if self.activation == 'softmax':
                assignments = torch.transpose(assignments, 1, 2).contiguous()
                assignments = assignments.view(self.batch_size*self.timesteps*self.in_shape[2]*self.in_shape[3], self.num_centers)
                assignments = torch.nn.functional.softmax(assignments) #my_softmax(assignments, dim=1)
                assignments = assignments.view(self.batch_size*self.timesteps, self.in_shape[2]*self.in_shape[3], self.num_centers)
                assignments = torch.transpose(assignments, 1, 2).contiguous()
            else:
                print('TODO implementation ...')
                exit()

        ## alpha *c 
        ## a_sum: batch_size, timesteps, num_centers, 1
        a_sum = torch.sum(assignments, -1, keepdim=True)

        ## a: batch_size*timesteps, num_centers, redu_dim
        a = a_sum * self.centers.view(1, self.num_centers, self.redu_dim)

        ## alpha* input_tensor
        ## fea_assign: batch_size, timesteps, num_centers, h, w ==> batch_size*timesteps, num_centers, h*w 
        # fea_assign = assignments.view(self.batch_size*self.timesteps, self.num_centers, self.in_shape[2]*self.in_shape[3])

        ## input_tensor: batch_size, timesteps, redu_dim, h, w  ==> batch_size*timesteps, redu_dim, h*w  ==>  batch_size*timesteps, h*w, redu_dim 
        input_tensor = input_tensor.view(self.batch_size*self.timesteps, self.redu_dim, self.in_shape[2]*self.in_shape[3])
        input_tensor = torch.transpose(input_tensor, 1, 2)

        ## x: batch_size*timesteps, num_centers, redu_dim
        x  = torch.matmul(assignments, input_tensor)


        ## batch_size*timesteps, num_centers, redu_dim
        vlad = x - a 

        ## batch_size*timesteps, num_centers, redu_dim ==> batch_size, timesteps, num_centers, redu_dim
        vlad = vlad.view(self.batch_size, self.timesteps, self.num_centers, self.redu_dim)
#         return vlad

        ## batch_size, num_centers, redu_dim 
        vlad = torch.sum(vlad, 1, keepdim=False)

        ## intor normalize
        vlad = torch.nn.functional.normalize(vlad, p=2, dim=2)

        ## l2-normalize
        vlad = vlad.view(self.batch_size, self.num_centers*self.redu_dim)
        vlad = torch.nn.functional.normalize(vlad, p=2, dim=1)

        # vlad = torch.Tensor([vlad]).cuda() # NEW line
        if not self.with_center_loss:
            return vlad
        else:
            assignments
            assignments = assignments.view(self.batch_size, self.timesteps, self.num_centers, self.in_shape[2]*self.in_shape[3])
            assign_predict = torch.sum(torch.sum(assignments, 3),1)
            return assign_predict, vlad

