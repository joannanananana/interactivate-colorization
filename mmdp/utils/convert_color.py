import torch


def lab2rgb(L, ab):
    lab = torch.cat((L * 100, ab * 255 - 127), dim=1)
    out = xyz2rgb(lab2xyz(lab))
    return out


def xyz2rgb(xyz):
    r = (
        3.24048134 * xyz[:, 0, :, :]
        - 1.53715152 * xyz[:, 1, :, :]
        - 0.49853633 * xyz[:, 2, :, :]
    )
    g = (
        -0.96925495 * xyz[:, 0, :, :]
        + 1.87599 * xyz[:, 1, :, :]
        + 0.04155593 * xyz[:, 2, :, :]
    )
    b = (
        0.05564664 * xyz[:, 0, :, :]
        - 0.20404134 * xyz[:, 1, :, :]
        + 1.05731107 * xyz[:, 2, :, :]
    )

    rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
    rgb = torch.max(
        rgb, torch.zeros_like(rgb)
    )  # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > 0.0031308).type(torch.FloatTensor)
    if rgb.is_cuda:
        mask = mask.cuda()

    rgb = (1.055 * (rgb ** (1.0 / 2.4)) - 0.055) * mask + 12.92 * rgb * (1 - mask)
    return rgb


def lab2xyz(lab):
    y_int = (lab[:, 0, :, :] + 16.0) / 116.0
    x_int = (lab[:, 1, :, :] / 500.0) + y_int
    z_int = y_int - (lab[:, 2, :, :] / 200.0)
    if z_int.is_cuda:
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat(
        (x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1
    )
    mask = (out > 0.2068966).type(torch.FloatTensor)
    if out.is_cuda:
        mask = mask.cuda()

    out = (out**3.0) * mask + (out - 16.0 / 116.0) / 7.787 * (1 - mask)

    sc = torch.Tensor((0.95047, 1.0, 1.08883))[None, :, None, None]
    sc = sc.to(out.device)

    out = out * sc
    return out
