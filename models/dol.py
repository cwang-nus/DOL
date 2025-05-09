import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.gcn import GCN
from layers.MLP import Convs1D

class DOL(nn.Module):
    def __init__(self,
                 args,
                 dropout=0.3,
                 supports_len=2,
                 gcn_bool=True,
                 addaptadj=True,
                 aptinit=None,
                 residual_channels=32,
                 dilation_channels=32,
                 skip_channels=256,
                 end_channels=512,
                 kernel_size=2,
                 blocks=4,
                 layers=2,
                 sample_type=None):
        super().__init__()

        self.sample_type = sample_type
        self.device = args.device

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.pred_len = args.pred_len
        self.node_num = args.node_num

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.instancenorm = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.residual_channels = residual_channels

        self.out_layers = nn.ModuleList()

        self.vi_adapter = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        for i in range(self.node_num):
            self.vi_adapter.append(Convs1D(residual_channels, residual_channels, args.lsa_dim, args.lsa_num, 0.5))

        receptive_field = 1

        self.supports_len = supports_len

        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(self.node_num, 10).to(
                    self.device), requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(torch.randn(10, self.node_num).to(
                    self.device), requires_grad=True).to(self.device)
                self.supports_len += 1
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(
                    initemb1, requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(
                    initemb2, requires_grad=True).to(self.device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(
                        GCN(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv = nn.Sequential(nn.Conv2d(in_channels=skip_channels,
                                               out_channels=end_channels,
                                                kernel_size=(1, 1),
                                                bias=True), nn.ReLU())

        self.decoder = nn.Conv2d(in_channels=end_channels,
                                   out_channels=self.pred_len*1,
                                   kernel_size=(1, 1),
                                   bias=True)

        self.receptive_field = receptive_field

    def forward(self, input, supports):

        bs, ts, n = input.shape
        input = input.unsqueeze(-1)

        input = input.permute(0, 3, 2, 1)  # b*dim*n*ts

        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(
                input, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = input

        x = self.start_conv(x)

        x_adaptive = torch.zeros([bs, self.residual_channels, n, x.size(-1)], dtype=x.dtype).to(x.device)

        for i in range(self.node_num):
            c_x = self.vi_adapter[i](x[:, :, i])
            x_adaptive[:, :, i] = c_x

        x = x + x_adaptive

        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj:
            adp = F.softmax(
                F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            residual = x

            # dilated convolution
            filter = self.filter_convs[i](residual)

            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)

            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        skip = skip[..., -1:]
        x = F.relu(skip)
        x = self.end_conv(x)

        x = self.decoder(x).squeeze(-1)

        return x
