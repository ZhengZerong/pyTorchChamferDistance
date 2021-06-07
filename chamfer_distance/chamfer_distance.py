
import torch

from torch.utils.cpp_extension import load
cd = load(name="cd",
          sources=["chamfer_distance/chamfer_distance.cpp",
                   "chamfer_distance/chamfer_distance.cu"],
          # extra_include_paths=["E:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include",
          #                      "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.18362.0\\shared",
          #                      "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.18362.0\\ucrt",
          #                      ],
          # extra_cflags=["-DWIN32", "-DWIN64"],
          # extra_cuda_cflags=["-DWIN32", "-DWIN64"],
          # extra_ldflags=["/NODEFAULTLIB:LIBCMT.LIB"],
          verbose=True)

class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        if not graddist1.is_cuda:
            cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2


class ChamferDistance(torch.nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)


class ChamferDistanceLoss(torch.nn.Module):
    def __init__(self, reduction=None):
        super(ChamferDistanceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, xyz1, xyz2, w1=None, w2=None):
        dist = list(ChamferDistanceFunction.apply(xyz1, xyz2))
        if w1 is not None:
            dist[0] = dist[0] * w1
        if w2 is not None:
            dist[1] = dist[1] * w2

        if self.reduction is None or self.reduction == "mean":
            return torch.mean(dist[0]) + torch.mean(dist[1])
        else:
            return torch.sum(dist[0]) + torch.sum(dist[1])
