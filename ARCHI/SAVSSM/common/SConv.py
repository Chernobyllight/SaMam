import torch
import torch.nn.functional as F
from math import ceil

class SConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,groups,representation_dim):
        super(SConv, self).__init__()
        reflection_padding = kernel_size // 2  # same dimension after padding
        self.reflection_padding = reflection_padding
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.kernel_size = kernel_size
        self.groups = groups

        padding = (kernel_size - 1) / 2
        self.compress_key = torch.nn.Sequential(
            torch.nn.Conv2d(representation_dim,
                                 out_channels,
                                 kernel_size=kernel_size,
                                 padding=(ceil(padding), ceil(padding)),
                                 padding_mode='reflect'),
                            torch.nn.LeakyReLU(0.1, True)
                                                )
    def forward(self, x,representation):
        # print(x.shape)
        out = self.reflection_pad(x)

        b, c, h, w = out.size()

        kernel = self.compress_key(representation).view(-1, 1, self.kernel_size, self.kernel_size)
        out = F.conv2d(out.view(1, -1, h, w), kernel, groups=b * c, padding=0)
        b,c,h,w = x.size()
        out = out.view(b, -1, h, w)

        # print(out.shape)
        # exit(123)

        return out


class SConv2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups,representation_dim):
        super(SConv2, self).__init__()
        reflection_padding = kernel_size // 2  # same dimension after padding
        self.reflection_padding = reflection_padding
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.kernel_size = kernel_size
        self.compress_key = torch.nn.Sequential(
            torch.nn.Linear(representation_dim, out_channels * kernel_size * kernel_size, bias=False),
            torch.nn.LeakyReLU(0.1, True)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
    def forward(self, x,representation):
        out = self.reflection_pad(x)
        b, c, h, w = out.size()
        kernel = self.compress_key(representation).view(b,self.out_channels, -1, self.kernel_size, self.kernel_size)
        # 1,64,1,kh,kw -> 1,64,4,kh,kw
        features_per_group = int(self.in_channels/self.groups)
        kernel = kernel.repeat_interleave(features_per_group, dim=2)
        # 1,64,4,kh,kw
        k_batch,k_outputchannel,k_feature_pergroup,kh,kw = kernel.size()
        out = F.conv2d(out.view(1, -1, h, w), kernel.view(-1,k_feature_pergroup,kh,kw), groups=b * self.groups, padding=0)
        b,c,h,w = x.size()
        out = out.view(b, -1, h, w)
        return out