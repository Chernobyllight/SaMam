

import torch.nn as nn

from ARCHI.SAVSSM.SS2D_Decoder import SS2D
from ARCHI.SAVSSM.common.SAIN import SRAdaIN
from ARCHI.SAVSSM.common.SCM import SCM

from ARCHI.archi_utils import PatchEmbed, PatchUnEmbed


class SAVSSM(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 64,
            d_state: int = 16,
            expand: float = 2.,
            representation_dim: int = 64,
            mamba_from_trion=1,
            zero_init=0,
            **kwargs,
    ):
        super().__init__()
        self.SAIN1 = SRAdaIN(in_channels=hidden_dim, representation_dim=representation_dim,zero_init=zero_init)
        self.SSM = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,
                        representation_dim=representation_dim,mamba_from_trion=mamba_from_trion,zero_init=zero_init,**kwargs)

        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()

        self.ca = SCM(representation_dim=representation_dim, channels_out=hidden_dim, reduction=4, zero_init=zero_init)

    def forward(self, input, representation):
        # x [B,HW,C]
        B, C, H, W = input.size()

        # branch 1
        x = self.SAIN1(input, representation) # x: B, C, H, W
        x = self.patch_embed(x)  # B,L,C
        # B, L, C = input.shape
        x = x.view(B, H, W, C).contiguous()  # x -> [B,H,W,C]
        x = self.SSM(x, representation)
        x = x.view(B, -1, C).contiguous()
        x = self.patch_unembed(x, C, H, W)

        # branch 2
        x1 = self.ca([input, representation])

        return x + x1





if __name__ == '__main__':
    import torch

    net = SAVSSM(hidden_dim=32,expand=2,representation_dim=32,mamba_from_trion=0).cuda()

    print('# net parameters:', sum(param.numel() for param in net.parameters()), '\n')

    img = torch.randn((1,32,20,35)).cuda()
    # print(img)
    degra = torch.randn((1,32,3,3)).cuda()
    out = net.forward(img,degra)
    print(out.shape)




