
import torch.nn as nn

from ARCHI.Patch_EmbedNN import PatchEmbedNN
from ARCHI.StyleEmbedder import GlobalStyleEncoder
from ARCHI.VSSM import VSSM
from ARCHI.SAVSSG import SAVSSG
from ARCHI.Decoder import Decoder_NN, Decoder_NN_x4
from ARCHI.LoE import LoE


class SaMam(nn.Module):
    def __init__(self, nVSSMs=2,
                       nSAVSSMs=2,
                       nSAVSSGs=2,
                 embed_dim=256, patch_size=8, representation_dim=64,d_state=16,expand=2.,
                 compress_ratio=8, squeeze_factor=8,mamba_from_trion=1):
        super().__init__()

        print('-----------init SaMam model-----------')
        if mamba_from_trion == 1:
            print("inference by mamba-ssm, quick")
        elif mamba_from_trion == 0:
            print("inference by torch, much more slower")



        content_encoder = [PatchEmbedNN(patch_size=patch_size, in_chans=3, embed_dim=embed_dim)]
        for _ in range(nVSSMs):
            content_encoder.append(VSSM(hidden_dim=embed_dim, d_state=d_state, expand=expand,mamba_from_trion=mamba_from_trion))
        content_encoder.append(LoE(num_feat=embed_dim, compress_ratio=compress_ratio,squeeze_factor=squeeze_factor))
        self.content_encoder = nn.Sequential(*content_encoder)

        style_encoder = [PatchEmbedNN(patch_size=patch_size, in_chans=3, embed_dim=embed_dim)]
        for _ in range(nVSSMs):
            style_encoder.append(
                VSSM(hidden_dim=embed_dim, d_state=d_state, expand=expand, mamba_from_trion=mamba_from_trion))
        style_encoder.append(LoE(num_feat=embed_dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor))
        self.style_encoder = nn.Sequential(*style_encoder)


        self.global_embedder = GlobalStyleEncoder(channels=embed_dim, representation_dim=representation_dim,embed_dim=embed_dim,patch_size=patch_size)


        SAVSSGs = [SAVSSG(
            nSAVSSMs=nSAVSSMs,hidden_dim=embed_dim, d_state=d_state, expand=expand, representation_dim=representation_dim,
                              compress_ratio=compress_ratio,squeeze_factor=squeeze_factor,mamba_from_trion=mamba_from_trion) \
                   for _ in range(nSAVSSGs)]
        self.SAVSSGs = nn.Sequential(*SAVSSGs)
        self.nSAVSSGs = nSAVSSGs

        if patch_size == 4:
            print("upsample: 4")
            self.decoder = Decoder_NN_x4(feature_channel=embed_dim)
        elif patch_size == 8:
            print("upsample: 8")
            self.decoder = Decoder_NN(feature_channel=embed_dim)
        else:
            print('patch size should be 4 or 8')
            exit()


    def forward(self, content, style):
        content_f = self.content_encoder(content)
        style_f = self.style_encoder(style)

        style_embedding = self.global_embedder(style_f)

        res = content_f
        for i in range(self.nSAVSSGs):
            res = self.SAVSSGs[i](res, style_embedding)
        Ics = self.decoder(res)

        return Ics

if __name__ == '__main__':
    import torch

    net = SaMam(embed_dim=256, patch_size=8, representation_dim=64,d_state=16,expand=2.,
                 compress_ratio=8, squeeze_factor=8,mamba_from_trion=1).cuda()

    print('# net parameters:', sum(param.numel() for param in net.parameters()), '\n')

    c = torch.randn((1,3,512,512)).cuda()
    s = torch.randn((1,3,512,512)).cuda()
    out = net.forward(c, s)
    print(out.shape)

    import torch
    from thop import profile

    flops, params = profile(net, (c,s))
    print('MACs: ', flops, 'params: ', params)

    print('MACs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

    model_para = sum(param.numel() for param in net.parameters())
    print('# MODEL parameters:'+ str(model_para/1000**2) + 'M', '\n')
