import torch.nn as nn

class PatchEmbedNN(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=8, in_chans=3, embed_dim=256):
        super().__init__()


        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        return x

if __name__ == '__main__':
    import torch

    net = PatchEmbedNN(patch_size=8, in_chans=3, embed_dim=256).cuda()

    print('# net parameters:', sum(param.numel() for param in net.parameters()), '\n')

    img = torch.randn((1,3,512,256)).cuda()
    out = net.forward(img)
    print(out.shape)
