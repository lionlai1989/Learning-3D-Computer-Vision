import argparse
import os
import time

import imageio
import numpy as np
import torch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

import dataset_location
import losses
from r2n2_custom import R2N2
from myutils import (
    get_device,
    render_mesh,
    render_pointcloud,
    # render_voxel,
    render_voxels_as_mesh,
)


def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(
        mesh_src.verts_packed().shape, requires_grad=True, device="cuda"
    )
    optimizer = torch.optim.Adam([deform_vertices_src], lr=args.lr)

    fitted_meshes = []

    for step in range(start_iter, args.max_iter + 1):
        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)
        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            # NOTE: The loss here should be a scalar value. Thus, using `item()` should
            # be enough to get the loss (`item()` only works for tensors with one
            # element). In my opinion, it doesn't make sense to call `detach()` here
            # because `detach()` returns a new Tensor and detached itself from the
            # current computation graph such that the autograd does not track it for
            # future backpropagations. `cpu()` returns a copy of itself in CPU memory.
            loss_vis = loss.cpu().item()
            print("[%4d/%4d]; loss: %.3f" % (step, args.max_iter, loss_vis))

            mesh_img = render_mesh(new_mesh_src, get_device())
            mesh_img *= 255
            fitted_meshes.append(mesh_img.astype(np.uint8))
    imageio.mimsave("./output_fit_data/mesh_fitted.gif", fitted_meshes, fps=5)

    mesh_src.offset_verts_(deform_vertices_src)

    # Save ground truth.
    mesh_img = render_mesh(mesh_tgt, "cuda:0")
    mesh_img *= 255
    imageio.imwrite("./output_fit_data/mesh_gt.png", mesh_img.astype(np.uint8))

    total_time = round(time.time() - start_time, 2)
    print(f"Done! Takes {total_time} seconds in total.")


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()
    optimizer = torch.optim.Adam([pointclouds_src], lr=args.lr)

    fitted_pointclouds = []

    for step in range(start_iter, args.max_iter + 1):
        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            # NOTE: The loss here should be a scalar value. Thus, using `item()` should
            # be enough to get the loss (`item()` only works for tensors with one
            # element). In my opinion, it doesn't make sense to call `detach()` here
            # because `detach()` returns a new Tensor and detached itself from the
            # current computation graph such that the autograd does not track it for
            # future backpropagations. `cpu()` returns a copy of itself in CPU memory.
            loss_vis = loss.cpu().item()
            print("[%4d/%4d]; loss: %.3f" % (step, args.max_iter, loss_vis))

            pc_img = render_pointcloud(
                pointclouds_src, image_size=128, device=get_device()
            )
            pc_img *= 255
            fitted_pointclouds.append(pc_img.astype(np.uint8))
    imageio.mimsave(
        "./output_fit_data/pointcloud_fitted.gif", fitted_pointclouds, fps=5
    )

    # Save ground truth.
    pc_img = render_pointcloud(pointclouds_tgt, image_size=128, device=get_device())
    pc_img *= 255
    imageio.imwrite("./output_fit_data/pointcloud_gt.png", pc_img.astype(np.uint8))

    total_time = round(time.time() - start_time, 2)
    print(f"Done! Takes {total_time} seconds in total.")


def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()
    optimizer = torch.optim.Adam([voxels_src], lr=args.lr)

    fitted_voxels = []

    for step in range(start_iter, args.max_iter + 1):
        # voxel_clipped = torch.nn.sigmoid(voxels_src)
        loss = losses.voxel_loss(voxels_src, voxels_tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            # NOTE: The loss here should be a scalar value. Thus, using `item()` should
            # be enough to get the loss (`item()` only works for tensors with one
            # element). In my opinion, it doesn't make sense to call `detach()` here
            # because `detach()` returns a new Tensor and detached itself from the
            # current computation graph such that the autograd does not track it for
            # future backpropagations. `cpu()` returns a copy of itself in CPU memory.
            loss_vis = loss.cpu().item()
            print("[%4d/%4d]; loss: %.3f" % (step, args.max_iter, loss_vis))
            voxel_img = render_voxels_as_mesh(voxels_src, "cuda:0", image_size=128)
            voxel_img *= 255
            fitted_voxels.append(voxel_img.astype(np.uint8))
    imageio.mimsave("./output_fit_data/voxel_fitted.gif", fitted_voxels, fps=5)

    # Save ground truth.
    voxel_img = render_voxels_as_mesh(voxels_tgt, "cuda:0", image_size=128)
    voxel_img *= 255
    voxel_img = voxel_img.astype(np.uint8)
    print("voxel_img: ", voxel_img.shape, voxel_img.dtype)
    imageio.imwrite("./output_fit_data/voxel_gt.png", voxel_img)

    total_time = round(time.time() - start_time, 2)
    print(f"Done! Takes {total_time} seconds in total.")


def train_model(args):
    r2n2_dataset = R2N2(
        "train",
        dataset_location.SHAPENET_PATH,
        dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH,
        return_voxels=True,
    )

    feed = r2n2_dataset[5]  # Use 0th image
    img = feed["images"].cpu().numpy()
    img *= 255
    imageio.imwrite("./output_fit_data/groundtruth_image.png", img.astype(np.uint8))

    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()

    if args.type == "vox":
        voxels_src = torch.rand(
            feed_cuda["voxels"].shape, requires_grad=True, device=args.device
        )
        # voxel_coords = feed_cuda["voxel_coords"].unsqueeze(0)
        voxels_tgt = feed_cuda["voxels"]
        print("voxels_src: ", voxels_src.shape)

        fit_voxel(voxels_src, voxels_tgt, args)

    elif args.type == "point":
        pointclouds_src = torch.randn(
            [1, args.n_points, 3], requires_grad=True, device=args.device
        )
        mesh_tgt = Meshes(verts=[feed_cuda["verts"]], faces=[feed_cuda["faces"]])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)

    elif args.type == "mesh":
        # Try different ways to initialize the source mesh.
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda["verts"]], faces=[feed_cuda["faces"]])

        fit_mesh(mesh_src, mesh_tgt, args)


def get_args_parser():
    parser = argparse.ArgumentParser("Model Fit", add_help=False)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_iter", default=50000, type=int)
    parser.add_argument("--type", choices=["vox", "point", "mesh"], type=str)
    parser.add_argument("--n_points", default=10000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=0.1, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    return parser


# python fit_data.py --type "vox"
# python fit_data.py --type "point"
# python fit_data.py --type "mesh"
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model Fit", parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
