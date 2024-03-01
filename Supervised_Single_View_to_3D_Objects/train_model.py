import argparse
import time

import torch
import wandb
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
from tqdm import tqdm

import dataset_location
import losses
from model_mesh import MeshModel
from model_point import PointModel
from model_vox import VoxModel
from r2n2_custom import R2N2

# Read https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy("file_system")

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


def get_args_parser():
    parser = argparse.ArgumentParser("Singleto3D", add_help=False)

    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--max_epoch", default=100, type=int)
    parser.add_argument("--log_freq", default=100, type=str)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--type", default="vox", choices=["vox", "point", "mesh"], type=str
    )
    parser.add_argument("--n_points", default=5000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=1.0, type=float)
    parser.add_argument("--save_freq", default=1000, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--load_feat", action="store_true")
    parser.add_argument("--load_checkpoint_epoch", default=None, type=int)
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
    images = feed_dict["images"].squeeze(1)  # (batch, 137, 137, 3)

    if args.type == "vox":
        voxels = feed_dict["voxels"].float()
        ground_truth_3d = voxels
    elif args.type == "point":
        mesh = feed_dict["mesh"].to(args.device)  # vertex is between -1 and 1
        # verts = mesh.verts_packed()
        # print("verts: ", verts.shape)
        # print("Center: ", verts.mean(dim=0))
        # print("Scale: ", verts.min(dim=0), "to", verts.max(dim=0))
        ground_truth_3d = sample_points_from_meshes(
            mesh, args.n_points
        )  # (batch, n_points, 3)
        # print(ground_truth_3d.is_cuda) # True
        return images.to(args.device), ground_truth_3d
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]

    # if args.load_feat:
    #     feats = torch.stack(feed_dict["feats"])
    #     return feats.to(args.device), ground_truth_3d.to(args.device)
    # else:
    #     return images.to(args.device), ground_truth_3d.to(args.device)


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

        prediction_3d = model(images_gt, args)  # (batch, n_points, 3)

        loss = calculate_loss(prediction_3d, ground_truth_3d, args)  # loss is scalar
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
        return_all_views=True,
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

    if args.type == "vox":
        model = VoxModel(args)
    elif args.type == "point":
        model = PointModel(args)
    elif args.type == "mesh":
        model = MeshModel(args)
    else:
        raise ValueError("Wrong model type.")
    model.to(args.device)
    model.train()
    print(f"model {model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # to use with ViTs
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    start_epoch = 0

    if args.load_checkpoint_epoch:
        checkpoint = torch.load(
            f"checkpoint_{args.type}_epoch_{args.load_checkpoint_epoch:03}.pth"
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Succesfully loaded epoch {start_epoch}")

    print("Starting training !")
    start_time = time.time()

    for epoch in range(start_epoch, args.max_epoch + 1):
        train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
        )
        lr_scheduler.step()

        if (epoch % 5) == 0:  # Save every 5 epochs.
            save_checkpoint(
                epoch,
                model,
                optimizer,
                lr_scheduler,
                f"checkpoint_{args.type}_epoch_{epoch:03}.pth",
            )
        torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time
    print(f"The code ran for {elapsed_time} seconds.")


# python3 train_model.py --type "point" --batch_size 32 --max_epoch 100 --num_workers 6 --arch resnet18 --n_points 10000
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
