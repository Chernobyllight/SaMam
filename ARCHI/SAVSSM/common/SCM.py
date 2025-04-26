import torch.nn as nn

class SCM(nn.Module):
    def __init__(self, representation_dim, channels_out, reduction, zero_init):
        super(SCM, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(representation_dim, representation_dim//reduction, 3, 3, 0, bias=False),
            nn.PReLU(),
            nn.Conv2d(representation_dim // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        if zero_init == 1:
            print('SCM zero init')
            for m in self.conv_du.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        att = self.conv_du(x[1])
        return x[0] * att


