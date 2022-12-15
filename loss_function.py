import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
from layers import Sobel


class GradientConv3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GradientConv3, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        kernel = [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ]

        kernel = torch.tensor(kernel).float().expand([out_dim, in_dim, 3, 3])
        # print('kernel_size:', kernel.shape)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        pad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))
        x1 = pad(x)
        x1 = F.conv2d(x1, self.weight)
        # print('x1.change:', x1.shape)

        return x1


class SinglePAM(nn.Module):

    def __init__(self, in_dim):
        super(SinglePAM, self).__init__()

        self.chanel_in = in_dim

        self.source_conv = GradientConv3(in_dim=in_dim, out_dim=in_dim)
        self.fuse_conv = GradientConv3(in_dim=in_dim, out_dim=in_dim)

        self.softmax = Softmax(dim=-1)

    def forward(self, x1, x2):
        m_batchsize, C, height, width = x1.size()

        x1 = self.source_conv(x1).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        x2 = self.fuse_conv(x2).view(m_batchsize, -1, width * height)

        energy = torch.bmm(x1, x2)

        attention = self.softmax(energy)
        # attention = torch.ones_like(attention) - attention

        return attention


class GradConsistencyLoss(nn.Module):

    def __init__(self, in_dim=1):
        super(GradConsistencyLoss, self).__init__()

        self.pam = SinglePAM(in_dim)

    def forward(self, a1, a2, fused):
        a1 = self.pam(a1, fused)
        a2 = self.pam(a2, fused)

        gc_loss = torch.mean(torch.pow((a1 - a2), 2))

        return gc_loss


class Edgeloss(nn.Module):

    def __init__(self, in_dim=1):
        super(Edgeloss, self).__init__()

        self.source_conv = Sobel(in_dim, in_dim)
        self.fuse_conv = Sobel(in_dim, in_dim)

    def forward(self, input_image, output_image):
        x1 = self.source_conv(input_image)
        x2 = self.fuse_conv(output_image)

        grad_loss = torch.mean(torch.pow((x1 - x2), 2))

        return grad_loss


