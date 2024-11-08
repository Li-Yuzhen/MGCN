import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def gravity_loss_2d(pred: Tensor, target: Tensor, n_classes):
    B, C, H, W = pred.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    place = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1).float().to(device)
    place_expanded = place.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1, -1)

    pred = F.softmax(pred, dim=1).float()
    target = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()

    sum1 = torch.sum(pred, dim=(2, 3)) + 1e-6  # torch.Size([B, C])

    pred_x = torch.mul(pred, torch.squeeze(place_expanded[..., 0:1]).to(device))
    pred_x = torch.sum(pred_x, dim=(2, 3))  # torch.Size([B, C])
    pred_x = torch.div(pred_x, sum1)  # torch.Size([B, C])

    pred_y = torch.mul(pred, torch.squeeze(place_expanded[..., 1:2]).to(device))
    pred_y = torch.sum(pred_y, dim=(2, 3))
    pred_y = torch.div(pred_y, sum1)

    a = torch.stack((pred_x, pred_y), dim=-1)  # torch.Size([B, C, 2])
    a = (a / H * 2) - 1

    sum2 = torch.sum(target, dim=(2, 3)) + 1e-6

    target_x = torch.mul(target, torch.squeeze(place_expanded[..., 0:1]).to(device))
    target_x = torch.sum(target_x, dim=(2, 3))
    target_x = torch.div(target_x, sum2)

    target_y = torch.mul(target, torch.squeeze(place_expanded[..., 1:2]).to(device))
    target_y = torch.sum(target_y, dim=(2, 3))
    target_y = torch.div(target_y, sum2)

    b = torch.stack((target_x, target_y), dim=-1)
    b = (b / W * 2) - 1

    criterion = nn.MSELoss()
    gravity_loss = criterion(a, b)

    return gravity_loss


def gravity_loss_3d(pred: Tensor, target: Tensor, n_classes):
    B, C, H, W, D = pred.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    place = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), torch.arange(D), indexing='ij'),  dim=-1).float().to(device)
    place_expanded = place.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1, -1, -1)

    pred = F.softmax(pred, dim=1).float()
    target = torch.squeeze(target)
    target = F.one_hot(target.to(torch.int64), n_classes).permute(0, 4, 1, 2, 3).float()

    sum1 = torch.sum(pred, dim=(2, 3, 4)) + 1e-6  # torch.Size([B, C])

    pred_x = torch.mul(pred, torch.squeeze(place_expanded[..., 0:1]).to(device))
    pred_x = torch.sum(pred_x, dim=(2, 3, 4))  # torch.Size([B, C])
    pred_x = torch.div(pred_x, sum1)  # torch.Size([B, C])
    pred_x = (pred_x / H * 2) - 1

    pred_y = torch.mul(pred, torch.squeeze(place_expanded[..., 1:2]).to(device))
    pred_y = torch.sum(pred_y, dim=(2, 3, 4))
    pred_y = torch.div(pred_y, sum1)
    pred_y = (pred_y / W * 2) - 1

    pred_z = torch.mul(pred, torch.squeeze(place_expanded[..., 2:3]).to(device))
    pred_z = torch.sum(pred_z, dim=(2, 3, 4))
    pred_z = torch.div(pred_z, sum1)
    pred_z = (pred_z / D * 2) - 1

    a = torch.stack((pred_x, pred_y, pred_z), dim=-1)  # torch.Size([B, C, 3])
    # a = (a / H * 2) - 1
    # print('a:', a.shape, a)  # torch.Size([6, 5, 3])

    sum2 = torch.sum(target, dim=(2, 3, 4)) + 1e-6

    target_x = torch.mul(target, torch.squeeze(place_expanded[..., 0:1]).to(device))
    target_x = torch.sum(target_x, dim=(2, 3, 4))
    target_x = torch.div(target_x, sum2)
    target_x = (target_x / H * 2) - 1

    target_y = torch.mul(target, torch.squeeze(place_expanded[..., 1:2]).to(device))
    target_y = torch.sum(target_y, dim=(2, 3, 4))
    target_y = torch.div(target_y, sum2)
    target_y = (target_y / W * 2) - 1

    target_z = torch.mul(target, torch.squeeze(place_expanded[..., 2:3]).to(device))
    target_z = torch.sum(target_z, dim=(2, 3, 4))
    target_z = torch.div(target_z, sum2)
    target_z = (target_z / D * 2) - 1

    b = torch.stack((target_x, target_y, target_z), dim=-1)
    # b = (b / W * 2) - 1
    # print('b:', b.shape, b)

    criterion = nn.MSELoss()
    gravity_loss = criterion(a, b)

    return gravity_loss