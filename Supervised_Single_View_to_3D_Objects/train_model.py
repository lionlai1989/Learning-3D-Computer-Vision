import argparse
import sys
import time

import matplotlib.pyplot as plt
import torch
import wandb
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
from tqdm import tqdm

import dataset_location
import losses
from model import SingleViewto3D
from r2n2_custom import R2N2

# Read https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy("file_system")


def get_args_parser():
    parser = argparse.ArgumentParser("Singleto3D", add_help=False)
    # Model parameters
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--lr", default=5e-2, type=float)
    parser.add_argument("--max_iter", default=10000, type=int)
    parser.add_argument("--num_epoch", default=2, type=int)
    parser.add_argument("--log_freq", default=100, type=str)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--type", default="vox", choices=["vox", "point", "mesh"], type=str
    )
    parser.add_argument("--n_points", default=5000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=0.1, type=float)
    parser.add_argument("--save_freq", default=1000, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--load_feat", action="store_true")
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--unit_test", default=False, type=bool)
    return parser


def save_checkpoint(epoch, model, optimizer, lr_scheduler, out_path):
    """
    Save the current of neural networks and the state of training.
    """

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        },
        out_path,
    )


def preprocess(feed_dict, args):
    images = feed_dict["images"].squeeze(1)
    if args.type == "vox":
        voxels = feed_dict["voxels"].float()
        ground_truth_3d = voxels
    elif args.type == "point":
        mesh = feed_dict["mesh"]
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)
        ground_truth_3d = pointclouds_tgt
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]
    if args.load_feat:
        feats = torch.stack(feed_dict["feats"])
        return feats.to(args.device), ground_truth_3d.to(args.device)
    else:
        return images.to(args.device), ground_truth_3d.to(args.device)


def calculate_loss(predictions, ground_truth, args):
    if args.type == "vox":
        # [256, 1, 32, 32, 32], [256, 1, 32, 32, 32]
        loss = losses.voxel_loss(predictions, ground_truth)
    elif args.type == "point":
        loss = losses.chamfer_loss(predictions, ground_truth)
    elif args.type == "mesh":
        sample_trg = sample_points_from_meshes(ground_truth, args.n_points)
        sample_pred = sample_points_from_meshes(predictions, args.n_points)

        loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
        loss_smooth = losses.smoothness_loss(predictions)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth
    return loss


def train_one_epoch(model, data_loader, optimizer, epoch, args):
    total_loss = 0
    num_batch = len(data_loader)
    for idx, data in enumerate(tqdm(data_loader)):
        images_gt, ground_truth_3d = preprocess(data, args)  # push data into gpu
        prediction_3d = model(images_gt, args)
        print("ground_truth_3d.shape: ", ground_truth_3d.shape)
        loss = calculate_loss(prediction_3d, ground_truth_3d, args)
        # loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() / num_batch  # loss is the average of the batch.

    print(f"Epoch {epoch}, loss: {total_loss}")


def train_model(args):
    r2n2_dataset = R2N2(
        "train",
        dataset_location.SHAPENET_PATH,
        dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH,
        return_voxels=True,
        return_feats=args.load_feat,
    )
    train_loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True,
        shuffle=True,  # False by default
    )

    # train_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.train()
    print(f"model {model}")

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # to use with ViTs
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        checkpoint = torch.load(f"checkpoint_{args.type}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_iter = checkpoint["epoch"]
        print(f"Succesfully loaded epoch {start_iter}")

    print("Starting training !")
    for epoch in range(0, args.num_epoch):
        train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
        )
        lr_scheduler.step()

        save_checkpoint(
            epoch, model, optimizer, lr_scheduler, f"checkpoint_{args.type}.pth"
        )
        torch.cuda.empty_cache()

    # for step in range(start_iter, args.max_iter):
    #     iter_start_time = time.time()

    #     if step % len(train_loader) == 0:  # restart after one epoch
    #         train_loader = iter(loader)

    #     read_start_time = time.time()

    #     feed_dict = next(train_loader)

    #     images_gt, ground_truth_3d = preprocess(feed_dict, args)
    #     read_time = time.time() - read_start_time

    #     prediction_3d = model(images_gt, args)

    #     loss = calculate_loss(prediction_3d, ground_truth_3d, args)
    #     loss = loss.mean()

    #     # print(f"prediction_3d shape {prediction_3d.shape}, ground_truth_3d shape {ground_truth_3d.shape}")
    #     # print(f"loss {loss}, loss shape {loss.shape}")

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     total_time = time.time() - start_time
    #     iter_time = time.time() - iter_start_time

    #     if (step % args.save_freq) == 0:
    #         torch.save(
    #             {
    #                 "step": step,
    #                 "model_state_dict": model.state_dict(),
    #                 "optimizer_state_dict": optimizer.state_dict(),
    #             },
    #             f"checkpoint_{args.type}.pth",
    #         )

    #     if (step % args.log_freq) == 0:
    #         loss_vis = loss.cpu().item()
    #         print(
    #             "[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f"
    #             % (step, args.max_iter, total_time, read_time, iter_time, loss_vis)
    #         )
    #         # wandb.log({"step": step, "loss": loss_vis})

    print("Done!")


# python3 train_model.py --type "vox" --arch resnet34 --batch_size 32 --num_epoch 50 --num_workers 6
# python3 train_model.py --type "point" --batch_size 32 --num_epoch 50 --num_workers 6
# python3 train_model.py --type "mesh" --batch_size 32 --num_epoch 50 --num_workers 6
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
    args = parser.parse_args()

    print(" ------------ starting scrip ------------ ")
    # wandb.init(project="L3D-HW2")

    train_model(args)
