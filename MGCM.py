import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvGRU import ConvGRUCell

class FeatureCenters(nn.Module):
    def __init__(self, n_classes, n_channels):
        super(FeatureCenters, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_classes, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        conv_output = self.conv(inputs)
        softmax_output = self.softmax(conv_output)

        feature_centers = []
        for i in range(1, softmax_output.size(1)):  # Exclude background class
            class_output = softmax_output[:, i, :, :].unsqueeze(1)
            class_center = torch.mean(class_output * inputs, dim=(2, 3), keepdim=True)
            feature_centers.append(class_center)

        return feature_centers, softmax_output


class GraphModel(nn.Module):
    def  __init__(self, N, chnn_in):
        super().__init__()
        self.n_node = N
        chnn = chnn_in
        self.C_wgt = nn.Conv2d(chnn*(N-1), (N-1), 1, groups=(N-1), bias=False)
        self.ConvGRU = ConvGRUCell(chnn, chnn, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        b, n, c, h, w = inputs.shape
        feat_s = [inputs[:,ii,:] for ii in range(self.n_node)]
        pred_s =[]
        for idx_node in range(self.n_node):
            h_t = feat_s[idx_node]
            h_t_m = h_t.repeat(1, self.n_node-1, 1, 1)
            h_n = torch.cat([feat_s[ii] for ii in range(self.n_node) if ii != idx_node], dim=1)
            msg = self._get_msg(h_t_m, h_n)
            m_t = torch.sum(msg.view(b, -1, c, h, w), dim=1)
            h_t = self.ConvGRU(m_t, h_t)
            base = feat_s[idx_node]
            pred_s.append(h_t*self.gamma+base)
        pred = torch.stack(pred_s).permute(1, 0, 2, 3, 4).contiguous()
        return pred

    def _get_msg(self, x1, x2):
        b, c, h, w = x1.shape
        wgt = self.C_wgt(x1 - x2).unsqueeze(1).repeat(1, c//(self.n_node-1), 1, 1, 1).view(b, c, h, w)
        out = x2 * torch.sigmoid(wgt)
        return out


class GraphReasoning(nn.Module):
    def __init__(self, chnn_in, n_iter, n_node):
        super().__init__()
        self.n_iter = n_iter
        self.n_node = n_node
        self.feature = FeatureCenters(n_node + 1, chnn_in)
        self.graph = GraphModel(self.n_node, chnn_in)
        self.conv = nn.Conv2d(n_node * chnn_in, chnn_in, kernel_size=3, padding=1)
    def _inn(self, Func, feat):
        feat = [fm.unsqueeze(1) for fm in feat]  # n (B, 1, C, 1, 1)
        feat = torch.cat(feat, 1)  # (B, n, C, 1, 1)
        for ii in range(self.n_iter):
            feat = Func(feat)
        feat1 = torch.split(feat, 1, 1)
        feat2 = [fm for fm in feat1]
        return feat2

    def forward(self, inputs):
        B, C, H, W = inputs.size()
        feat, softmax_output = self.feature(inputs)
        feat = self._inn(self.graph, feat)
        feat = torch.cat(feat, 1)  # (B, n, C, 1, 1)
        expanded_feat = feat.expand(-1, -1, -1, H, W)  # (B, n, C, H, W)
        reshaped_feat = self.conv(expanded_feat.view(B, self.n_node * C, H, W))  # (B, C, H, W)
        x = torch.cat([inputs, reshaped_feat], dim=1)
        return x, softmax_output