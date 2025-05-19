

import torch.nn as nn

from ARCHI.SAVSSM.SAVSSM import SAVSSM
from ARCHI.LoE import LoE

class SAVSSG(nn.Module):
    def __init__(
            self,
            nSAVSSMs: int = 2,
            hidden_dim: int = 64,
            d_state: int = 16,
            expand: float = 2.,
            representation_dim: int = 64,
            compress_ratio=4, squeeze_factor=16,
            mamba_from_trion=1,
            zero_init=0,
    ):
        super().__init__()

        SAVSSMs = [SAVSSM(
            hidden_dim=hidden_dim, d_state=d_state, expand=expand,
            representation_dim=representation_dim, mamba_from_trion=mamba_from_trion,zero_init=zero_init) \
            for _ in range(nSAVSSMs)]
        self.SAVSSMs = nn.Sequential(*SAVSSMs)
        self.nSAVSSMs = nSAVSSMs


        self.LoE = LoE(num_feat=hidden_dim, compress_ratio=compress_ratio,squeeze_factor=squeeze_factor)

    def forward(self, input, representation):

        res = input
        for i in range(self.nSAVSSMs):
            res = self.SAVSSMs[i](res, representation)
        # x = self.SAVSSM1(input, representation)
        # x = self.SAVSSM2(x, representation)

        y = res + input

        y = self.LoE(y)

        return y





if __name__ == '__main__':
    import torch

    net = SAVSSG(hidden_dim=32,expand=2,representation_dim=32).cuda()

    print('# net parameters:', sum(param.numel() for param in net.parameters()), '\n')

    img = torch.randn((1,32,20,35)).cuda()
    # print(img)
    degra = torch.randn((1,32,3,3)).cuda()
    out = net.forward(img,degra)
    print(out.shape)




