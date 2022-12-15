import torch
import torch.nn as nn
from layers import Cutin, St_branch, SobelBranch, RSBranch


class FLFuseNet(nn.Module):
    def __init__(self):
        super(FLFuseNet, self).__init__()

        self.conva1 = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1, 1, bias=False), nn.ReLU())
        self.convb1 = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1, 1, bias=False), nn.ReLU())

        self.cuta1 = Cutin(8, 8)

        self.cuta2 = Cutin(16, 8)

        self.cuta3 = Cutin(24, 8)

        self.cuta4 = Cutin(32, 8)

        self.fusionlayer = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1, 1, bias=False), nn.ReLU())

        self.SobelBrancha = SobelBranch(1, 8)


        self.de = nn.Sequential(
            nn.Conv2d(24, 1, 3, 1, 1, 1, bias=False)
        )

    def forward(self, x1, x2):
        conv_a_out = self.conva1(x1)
        conv_b_out = self.convb1(x2)

        cut_a_1_out = self.cuta1(conv_a_out)
        cut_b_1_out = self.cuta1(conv_b_out)

        cut_a_2_in = torch.cat([cut_a_1_out, conv_a_out], 1)
        cut_b_2_in = torch.cat([cut_b_1_out, conv_b_out], 1)

        cut_a_2_out = self.cuta2(cut_a_2_in)
        cut_b_2_out = self.cuta2(cut_b_2_in)

        cut_a_3_in = torch.cat([conv_a_out, cut_a_1_out, cut_a_2_out], 1)
        cut_b_3_in = torch.cat([conv_b_out, cut_b_1_out, cut_b_2_out], 1)

        cut_a_3_out = self.cuta3(cut_a_3_in)
        cut_b_3_out = self.cuta3(cut_b_3_in)

        cut_a_4_in = torch.cat([conv_a_out, cut_a_1_out, cut_a_2_out, cut_a_3_out], 1)
        cut_b_4_in = torch.cat([conv_b_out, cut_b_1_out, cut_b_2_out, cut_b_3_out], 1)

        cut_a_4_out = self.cuta4(cut_a_4_in)
        cut_b_4_out = self.cuta4(cut_b_4_in)

        de_in = torch.cat([cut_a_4_out, cut_b_4_out], 1)

        fusion = self.fusionlayer(de_in)

        out = torch.cat([fusion,
                         self.SobelBrancha(x1),
                         # self.SobelBranchb(x2),
                         ], 1)

        out = self.de(out)

        return out



class FLFuseNet_NoEdgeBranch(nn.Module):
    def __init__(self):
        super(FLFuseNet_NoEdgeBranch, self).__init__()

        self.conva1 = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1, 1, bias=False), nn.ReLU())
        self.convb1 = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1, 1, bias=False), nn.ReLU())

        self.cuta1 = Cutin(8, 8)

        self.cuta2 = Cutin(16, 8)

        self.cuta3 = Cutin(24, 8)

        self.cuta4 = Cutin(32, 8)

        self.fusionlayer = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1, 1, bias=False), nn.ReLU())

        # self.SobelBrancha = SobelBranch(1, 8)


        self.de = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1, 1, bias=False)
        )

    def forward(self, x1, x2):
        conv_a_out = self.conva1(x1)
        conv_b_out = self.convb1(x2)

        cut_a_1_out = self.cuta1(conv_a_out)
        cut_b_1_out = self.cuta1(conv_b_out)

        cut_a_2_in = torch.cat([cut_a_1_out, conv_a_out], 1)
        cut_b_2_in = torch.cat([cut_b_1_out, conv_b_out], 1)

        cut_a_2_out = self.cuta2(cut_a_2_in)
        cut_b_2_out = self.cuta2(cut_b_2_in)

        cut_a_3_in = torch.cat([conv_a_out, cut_a_1_out, cut_a_2_out], 1)
        cut_b_3_in = torch.cat([conv_b_out, cut_b_1_out, cut_b_2_out], 1)

        cut_a_3_out = self.cuta3(cut_a_3_in)
        cut_b_3_out = self.cuta3(cut_b_3_in)

        cut_a_4_in = torch.cat([conv_a_out, cut_a_1_out, cut_a_2_out, cut_a_3_out], 1)
        cut_b_4_in = torch.cat([conv_b_out, cut_b_1_out, cut_b_2_out, cut_b_3_out], 1)

        cut_a_4_out = self.cuta4(cut_a_4_in)
        cut_b_4_out = self.cuta4(cut_b_4_in)

        de_in = torch.cat([cut_a_4_out, cut_b_4_out], 1)

        fusion = self.fusionlayer(de_in)

        # out = torch.cat([fusion,
        #                  self.SobelBrancha(x1),
        #                  # self.SobelBranchb(x2),
        #                  ], 1)

        out = self.de(fusion)

        return out
