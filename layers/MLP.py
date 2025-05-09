import torch.nn as nn

class Convs1D(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, mlp_width, mlp_depth, mlp_dropout, act=nn.ReLU()):
        super(Convs1D, self).__init__()

        if mlp_depth == 1:
            mlp_width = n_outputs

        self.input = nn.Conv1d(in_channels=n_inputs,
                               out_channels=mlp_width,
                               kernel_size=1)
        self.dropout = nn.Dropout(mlp_dropout)

        self.mlp_depth = mlp_depth

        if mlp_depth > 1:
            self.output = nn.Conv1d(in_channels=mlp_width,
                                    out_channels=n_outputs,
                                    kernel_size=1)
        if mlp_depth > 2:
            self.hiddens = nn.ModuleList([
                nn.Conv1d(in_channels=mlp_width,
                          out_channels=mlp_width,
                          kernel_size=1)
                for _ in range(mlp_depth-2)])

        self.n_outputs = n_outputs
        self.act = act

    def forward(self, x):
        x = self.input(x)
        if self.training:
            x = self.dropout(x)

        if self.mlp_depth > 1:
            x = self.act(x)

        if self.mlp_depth > 2:
            for hidden in self.hiddens:
                x = hidden(x)
                if self.training:
                    x = self.dropout(x)
                x = self.act(x)

        if self.mlp_depth > 1:
            x = self.output(x)
        return x

class Convs(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, mlp_width, mlp_depth, mlp_dropout, act=nn.ReLU()):
        super(Convs, self).__init__()

        self.mlp_depth = mlp_depth

        if mlp_depth == 1:
            mlp_width = n_outputs

        self.input = nn.Conv2d(in_channels=n_inputs,
                               out_channels=mlp_width,
                               kernel_size=(1, 1))
        self.dropout = nn.Dropout(mlp_dropout)

        if mlp_depth > 2:
            self.hiddens = nn.ModuleList([
                nn.Conv2d(in_channels=mlp_width,
                          out_channels=mlp_width,
                          kernel_size=(1, 1))
                for _ in range(mlp_depth-2)])

        if mlp_depth > 1:
            self.output = nn.Conv2d(in_channels=mlp_width,
                                    out_channels=n_outputs,
                                    kernel_size=(1, 1))

        self.n_outputs = n_outputs
        self.act = act

    def forward(self, x):
        x = self.input(x)
        if self.training:
            x = self.dropout(x)
        if self.mlp_depth > 1:
            x = self.act(x)
        if self.mlp_depth > 2:
            for hidden in self.hiddens:
                x = hidden(x)
                if self.training:
                    x = self.dropout(x)
                x = self.act(x)
        if self.mlp_depth > 1:
            x = self.output(x)
        return x