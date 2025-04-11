
import torch.nn as nn


class GlobalStyleEncoder(nn.Module):
    def __init__(self, channels, representation_dim, embed_dim):
        super().__init__()
        self.representation_dim = representation_dim
        self.downscale = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
            #
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
            #
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
        )
        self.globalPooling = nn.AdaptiveAvgPool2d((4, 4))
        in_features = embed_dim * 4 * 4
        out_features = representation_dim * 3 * 3
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, xs):
        ys = self.downscale(xs)
        if ys.shape[2] != 4 and ys.shape[3] != 4:
            ys = self.globalPooling(ys)
        ys = ys.reshape(len(xs), -1)
        w = self.fc(ys)
        w = w.reshape(len(xs), self.representation_dim, 3, 3)

        return w

if __name__ == '__main__':
    import torch

    net = GlobalStyleEncoder(channels=256, representation_dim=64, embed_dim=256).cuda()

    print('# net parameters:', sum(param.numel() for param in net.parameters()), '\n')

    img = torch.randn((1,256,64,64)).cuda()
    out = net.forward(img)
    print(out.shape)
