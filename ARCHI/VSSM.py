
import torch.nn as nn
from ARCHI.archi_utils import PatchEmbed,PatchUnEmbed


from ARCHI.SS2D_Encoder import SS2D_encoder


class VSSM(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 64,
            d_state: int = 16,
            expand: float = 2.,
            mamba_from_trion=1,
            **kwargs,
    ):
        super().__init__()
        self.SSM = SS2D_encoder(d_model=hidden_dim, d_state=d_state,expand=expand,mamba_from_trion=mamba_from_trion, **kwargs)

        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()

    def forward(self, input):
        # x [B,HW,C]
        B, C, H, W = input.size()

        # branch 1
        x = self.patch_embed(input)  # B,L,C
        # B, L, C = input.shape
        x = x.view(B, H, W, C).contiguous()  # x -> [B,H,W,C]
        x = self.SSM(x)
        x = x.view(B, -1, C).contiguous()
        x = self.patch_unembed(x, C, H, W)

        return x + input



if __name__ == '__main__':
    import torch

    net = VSSM(hidden_dim=32,expand=2,representation_dim=32).cuda()

    print('# net parameters:', sum(param.numel() for param in net.parameters()), '\n')

    img = torch.randn((1,32,20,35)).cuda()
    # print(img)
    degra = torch.randn((1,32)).cuda()
    out = net.forward(img)
    print(out.shape)






