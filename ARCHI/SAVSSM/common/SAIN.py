import torch
import torch.nn as nn

class SRAdaIN(torch.nn.Module):
    def __init__(self, in_channels,representation_dim, zero_init):
        super(SRAdaIN, self).__init__()
        self.inns = torch.nn.InstanceNorm2d(in_channels, affine=False)
        self.compress_gamma = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(representation_dim,
                      in_channels,
                      kernel_size=1),
            nn.LeakyReLU(0.1, True)
        )
        self.compress_beta = torch.nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(representation_dim,
                      in_channels,
                      kernel_size=1),
            nn.LeakyReLU(0.1, True)
        )

        if zero_init == 1:
            print('SAIN zero init')
            for m in self.compress_gamma.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            for m in self.compress_beta.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


    def forward(self, x, representation):
        # print(representation.shape)
        gamma = self.compress_gamma(representation)
        beta = self.compress_beta(representation)
        out = self.inns(x)
        out = out * gamma + beta
        return out

