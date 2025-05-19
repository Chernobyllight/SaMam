
import torch.nn as nn



class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence
    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, dim, H, W):
        x = x.transpose(1, 2).view(x.shape[0], dim, H, W)  # b Ph*Pw c
        return x


class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, embed_dim=96, norm_layer=None):
        super().__init__()
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x







def get_permute_order(h,w):
    H, W = h,w
    L = H * W

    # [start, right, left, up, down] [0, 1, 2, 3, 4]

    o1 = []
    d1 = []
    o1_inverse = [-1 for _ in range(L)]
    i, j = 0, 0
    j_d = "right"
    while i < H:
        assert j_d in ["right", "left"]
        idx = i * W + j
        o1_inverse[idx] = len(o1)
        o1.append(idx)
        if j_d == "right":
            if j < W - 1:
                j = j + 1
                d1.append(1)
            else:
                i = i + 1
                d1.append(4)
                j_d = "left"

        else:
            if j > 0:
                j = j - 1
                d1.append(2)
            else:
                i = i + 1
                d1.append(4)
                j_d = "right"
    d1 = [0] + d1[:-1]



    o2 = []
    d2 = []
    o2_inverse = [-1 for _ in range(L)]

    i, j = H - 1, W - 1
    j_d = "left"

    while i > -1:
        assert j_d in ["right", "left"]
        idx = i * W + j
        o2_inverse[idx] = len(o2)
        o2.append(idx)
        if j_d == "right":
            if j < W - 1:
                j = j + 1
                d2.append(1)
            else:
                i = i - 1
                d2.append(3)
                j_d = "left"
        else:
            if j > 0:
                j = j - 1
                d2.append(2)
            else:
                i = i - 1
                d2.append(3)
                j_d = "right"
    d2 = [0] + d2[:-1]



    o3 = []
    d3 = []
    o3_inverse = [-1 for _ in range(L)]
    i, j = 0, W-1
    i_d = "down"
    while j > -1:
        assert i_d in ["down", "up"]
        idx = i * W + j
        o3_inverse[idx] = len(o3)
        o3.append(idx)
        if i_d == "down":
            if i < H - 1:
                i = i + 1
                d3.append(4)
            else:
                j = j - 1
                d3.append(1)
                i_d = "up"
        else:
            if i > 0:
                i = i - 1
                d3.append(3)
            else:
                j = j - 1
                d3.append(1)
                i_d = "down"
    d3 = [0] + d3[:-1]


    o4 = []
    d4 = []
    o4_inverse = [-1 for _ in range(L)]



    i, j = H-1, 0
    i_d = "up"
    while j < W:
        assert i_d in ["down", "up"]
        idx = i * W + j
        o4_inverse[idx] = len(o4)
        o4.append(idx)
        if i_d == "down":
            if i < H - 1:
                i = i + 1
                d4.append(4)
            else:
                j = j + 1
                d4.append(2)
                i_d = "up"
        else:
            if i > 0:
                i = i - 1
                d4.append(3)
            else:
                j = j + 1
                d4.append(2)
                i_d = "down"
    d4 = [0] + d4[:-1]

    o1 = tuple(o1)
    d1 = tuple(d1)
    o1_inverse = tuple(o1_inverse)

    o2 = tuple(o2)
    d2 = tuple(d2)
    o2_inverse = tuple(o2_inverse)

    o3 = tuple(o3)
    d3 = tuple(d3)
    o3_inverse = tuple(o3_inverse)

    o4 = tuple(o4)
    d4 = tuple(d4)
    o4_inverse = tuple(o4_inverse)

    return (o1, o2, o3, o4), (o1_inverse, o2_inverse, o3_inverse, o4_inverse), (d1, d2, d3, d4)

if __name__ == '__main__':
    (o1, o2, o3, o4), (o1_inverse, o2_inverse, o3_inverse, o4_inverse), (d1, d2, d3, d4) = get_permute_order(3,2)
    print(o1)
    print(o1_inverse)
    print(d1)
    print('-----------------------')
    print(o2)
    print(o2_inverse)
    print(d2)
    print('-----------------------')
    print(o3)
    print(o3_inverse)
    print('-----------------------')
    print(o4)
    print(o4_inverse)





