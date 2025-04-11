import torch.nn as nn


class Decoder_NN(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, feature_channel):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel, feature_channel//2, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel//2, feature_channel//2, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel//2, feature_channel//2, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel//2, feature_channel//2, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel//2, feature_channel//4, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel//4, feature_channel//4, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel//4, feature_channel//8, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel//8, feature_channel//8, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel//8, 3, (3, 3)),
        )

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.decoder(x)
        return x




class Decoder_NN_x4(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, feature_channel):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel, feature_channel//2, (3, 3)),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel//2, feature_channel//2, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel//2, feature_channel//2, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel//2, feature_channel//4, (3, 3)),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel//4, feature_channel//4, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel//4, feature_channel//8, (3, 3)),
            nn.ReLU(),

            # nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel//8, feature_channel//8, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel//8, 3, (3, 3)),
        )

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.decoder(x)
        return x






if __name__ == '__main__':
    import torch

    net = Decoder_NN_x4(feature_channel=256).cuda()

    print('# net parameters:', sum(param.numel() for param in net.parameters()), '\n')

    img = torch.randn((1,256,48,48)).cuda()
    out = net.forward(img)
    print(out.shape)



