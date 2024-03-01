import torch
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.ops.knn import knn_points


def voxel_loss(voxel_src, voxel_tgt):
    # voxel_src, voxel_tgt: batch x height x width x depth
    # Each value in voxel_tgt is either 1 or 0, meaning it occupies this voxel or not.
    # This is called binary voxel grid.
    #
    # Explain the differences between:
    # `torch.nn.functional.binary_cross_entropy` and `torch.nn.BCELoss`
    # `torch.nn.functional.binary_cross_entropy_with_logits` and `torch.nn.BCEWithLogitsLoss`
    #
    # TL;DR: Use BCEWithLogitsLoss for numerical stability.
    # Read https://github.com/pytorch/pytorch/issues/751
    # NOTE: Internally `BCEWithLogitsLoss` applies sigmoid function to input but NOT on target.
    loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
    return loss(voxel_src, voxel_tgt)  # Returns a scalar Tensor.


def chamfer_loss(point_cloud_src, point_cloud_tgt):
    # point_cloud_src, point_cloud_src: batch x n_points x 3
    # Chamfer Distance (CD) is a evaluation metric for two point clouds.
    # Read https://github.com/UM-ARM-Lab/Chamfer-Distance-API

    batch_size, n_points, dim = point_cloud_src.shape
    assert batch_size == point_cloud_tgt.shape[0]
    assert n_points == point_cloud_tgt.shape[1]
    assert dim == 3 and dim == point_cloud_tgt.shape[2]

    k = 1  # the number of nearest neighbors
    # knn_points returns the squared distances to the K-Nearest neighbors on point clouds.
    # src_dists, tgt_dists: (batch x n_points x k)
    src_dists, _, _ = knn_points(point_cloud_src, point_cloud_tgt, K=k)
    tgt_dists, _, _ = knn_points(point_cloud_tgt, point_cloud_src, K=k)

    return src_dists.mean() + tgt_dists.mean()  # Returns a scalar Tensor.


def smoothness_loss(mesh_src):
    # https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html
    # It seems `mesh_laplacian_smoothing` already averages the loss by the size of the batch.
    return mesh_laplacian_smoothing(mesh_src)  # Returns a scalar Tensor.
