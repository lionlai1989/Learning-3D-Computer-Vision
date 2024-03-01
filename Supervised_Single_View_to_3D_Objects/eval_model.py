import argparse
import math
import sys
import time
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.transforms import Rotate, axis_angle_to_matrix
from tqdm import tqdm

import dataset_location
from model_mesh import MeshModel
from model_point import PointModel
from model_vox import VoxModel
from myutils import (
    Mem2Ref,
    get_device,
    render_mesh,
    render_pointcloud,
    render_rotating_meshes,
    render_rotating_pointclouds,
    render_rotating_voxels,
    render_voxels_as_mesh,
)
from r2n2_custom import R2N2

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

EVAL_DIR = Path("./evaluation")


def preprocess(feed_dict, args):
    for k in ["images"]:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict["images"].squeeze(1)
    mesh = feed_dict["mesh"]
    if args.load_feat:
        images = torch.stack(feed_dict["feats"]).to(args.device)

    return images, mesh


def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker="o")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1-score")
    ax.set_title(f"Evaluation {args.type}")
    plt.savefig(f"eval_{args.type}", bbox_inches="tight")


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],),
        pred_points.shape[1],
        dtype=torch.int64,
        device=pred_points.device,
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],),
        gt_points.shape[1],
        dtype=torch.int64,
        device=gt_points.device,
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(
        pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1
    )
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(
        gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1
    )
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics


def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions
        H, W, D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(
            voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5
        )
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = Mem2Ref(pred_points, H, W, D)
        # Apply a rotation transform to align predicted voxels to gt mesh
        angle = -math.pi
        axis_angle = torch.as_tensor(np.array([[0.0, angle, 0.0]]))
        Rot = axis_angle_to_matrix(axis_angle)
        T_transform = Rotate(Rot)
        pred_points = T_transform.transform_points(pred_points)
        # re-center the predicted points
        pred_points = pred_points - pred_points.mean(1, keepdim=True)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)

    if args.type == "vox":
        gt_points = gt_points - gt_points.mean(1, keepdim=True)
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics


def evaluate_model(args):
    r2n2_dataset = R2N2(
        "test",
        dataset_location.SHAPENET_PATH,
        dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH,
        return_voxels=True,
        return_feats=args.load_feat,
    )

    eval_loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True,
    )

    if args.type == "vox":
        model = VoxModel(args)
    elif args.type == "point":
        model = PointModel(arch=args.arch, n_points=args.n_points)
    elif args.type == "mesh":
        model = MeshModel(args)
    else:
        raise ValueError
    model.to(args.device)
    model.eval()

    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint_epoch:
        checkpoint_path = (
            f"checkpoint_{args.type}_epoch_{args.load_checkpoint_epoch:03}.pth"
        )
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Successfully load {checkpoint_path}")

    num_batch = len(eval_loader)
    print(f"Starting evaluating {num_batch} batches.")

    for index, feed_dict in enumerate(tqdm(eval_loader)):

        iter_start_time = time.time()
        read_start_time = time.time()

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)

        if (index % args.vis_freq) == 0:
            # Save input image
            img = images_gt.cpu().numpy()[0, ...]
            img *= 255
            imageio.imwrite(f"./evaluation/image_{index}.png", img.astype(np.uint8))

            # if args.type == "vox":
            #     img = render_rotating_voxels(
            #         predictions.squeeze(0), get_device(), image_size=512
            #     )
            #     imageio.mimsave(EVAL_DIR / f"voxel_eval_{index}.gif", img, fps=12)
            # elif args.type == "point":

            img = render_rotating_pointclouds(
                points=predictions,
                image_size=274,
                device=get_device(),
                background_color=(0, 0, 0),
                radius=0.006,
                dist=1.5,
            )
            imageio.mimsave(EVAL_DIR / f"pc_eval_{index}.gif", img, fps=12)

            img = render_rotating_pointclouds(
                points=sample_points_from_meshes(mesh_gt, args.n_points).to(
                    get_device()
                ),
                image_size=274,
                device=get_device(),
                background_color=(0, 0, 0),
                radius=0.006,
                dist=1.5,
            )
            imageio.mimsave(EVAL_DIR / f"pc_gt_{index}.gif", img, fps=12)

            # elif args.type == "mesh":
            #     # imgs = render_rotating_meshes(
            #     #     meshes=predictions, image_size=512, device=get_device()
            #     # )
            #     imgs = render_turntable_mesh(predictions, image_size=256, device=None)
            #     imageio.mimsave(EVAL_DIR / f"mesh_eval_{index}.gif", imgs, fps=12)

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics["F1@0.050000"]
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(
            torch.tensor([metrics["Precision@%f" % t] for t in thresholds])
        )
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        # print(# f-string outputs "TypeError: unsupported format string passed to Tensor.__format__"
        #     "[%-4d/%-4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" %
        #     (index, num_batch, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean().item())
        # )

    avg_f1_score = torch.stack(avg_f1_score).mean(0)
    save_plot(thresholds, avg_f1_score, args)

    print("Done!")


def get_args_parser():
    parser = argparse.ArgumentParser("Singleto3D", add_help=False)
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--vis_freq", default=1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=str)
    parser.add_argument(
        "--type", default="vox", choices=["vox", "point", "mesh"], type=str
    )
    parser.add_argument("--n_points", default=5000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=0.1, type=float)
    parser.add_argument("--load_checkpoint_epoch", default=None, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--load_feat", action="store_true")
    parser.add_argument("--unit_test", default=False, type=bool)
    return parser


# python3 eval_model.py --type "point" --load_checkpoint_epoch 100 --arch resnet18 --n_points 10000 --vis_freq 100
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
