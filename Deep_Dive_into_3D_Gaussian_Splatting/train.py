import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from model import Scene, Gaussians
from torch.utils.data import DataLoader
from data_utils import CowDataset, visualize_renders
from data_utils_harder_scene import get_nerf_datasets, trivial_collate

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import gc


def make_trainable(gaussians):
    # Find all relevant attributes of gaussians and make them trainable.
    attrs = ["means", "pre_act_scales", "colours", "pre_act_opacities"]
    if not gaussians.is_isotropic:
        print("3D Gaussians are anisotropic.")
        # If anisotropic, perform gradient descent on `pre_act_quats`.
        attrs += ["pre_act_quats"]

    # TODO: spherical_harmonics
    attrs += ["spherical_harmonics"]

    for attr in attrs:
        param = getattr(gaussians, attr)
        param.requires_grad_(True)


def setup_optimizer(gaussians):
    """Modify the learning rates to reasonable values. We have intentionally
    set very high learning rates for all parameters. Consider reducing the learning rates for parameters that seem to vary too
    fast with the default settings. Consider setting different learning rates for different sets of parameters.
    """
    gaussians.check_if_trainable()

    parameters = [
        {"params": [gaussians.pre_act_opacities], "lr": 5e-3, "name": "opacities"},
        {"params": [gaussians.pre_act_scales], "lr": 5e-3, "name": "scales"},
        {"params": [gaussians.colours], "lr": 5e-3, "name": "colours"},
        {"params": [gaussians.means], "lr": 1e-3, "name": "means"},
    ]
    optimizer = torch.optim.Adam(parameters, lr=1e-3, eps=1e-15)
    return optimizer


def run_training(args):
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    # Setting up dataset
    if args.dataset_name == "cow":
        train_dataset = CowDataset(root=args.data_path, split="train")
        test_dataset = CowDataset(root=args.data_path, split="test")
        collate_fn = CowDataset.collate_fn
    elif args.dataset_name == "materials":
        train_dataset, test_dataset, _ = get_nerf_datasets(
            dataset_name="materials",
            data_root=args.data_path,
            image_size=[128, 128],
        )
        collate_fn = trivial_collate
    else:
        raise ValueError(f"Unrecognize dataset {args.dataset_name}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn,
    )
    train_itr = iter(train_loader)

    # Preparing some code for visualization
    viz_gif_path_1 = os.path.join(args.out_path, "q1_training_progress.gif")
    viz_gif_path_2 = os.path.join(args.out_path, "q1_training_final_renders.gif")
    viz_idxs = np.linspace(0, len(train_dataset) - 1, 5).astype(np.int32)[:4]

    gt_viz_imgs = [
        (train_dataset[i][0] * 255.0).numpy().astype(np.uint8) for i in viz_idxs
    ]
    gt_viz_imgs = [np.array(Image.fromarray(x).resize((256, 256))) for x in gt_viz_imgs]
    gt_viz_img = np.concatenate(gt_viz_imgs, axis=1)

    viz_cameras = [train_dataset[i][1].cuda() for i in viz_idxs]

    # Init gaussians and scene. Do note that we are setting isotropic to True
    gaussians = Gaussians(
        load_path=train_dataset.points_path,
        init_type=args.init_type,
        device=args.device,
        isotropic=args.isotropic,
    )
    scene = Scene(gaussians)
    print(f"There are {len(gaussians)} Gaussians points.")

    # Making gaussians trainable and setting up optimizer
    make_trainable(gaussians)
    optimizer = setup_optimizer(gaussians)

    # Training loop
    viz_frames = []
    mse_loss = torch.nn.MSELoss(reduction="mean")
    # criterion = L1Loss()
    for itr in range(args.num_itrs):
        # Fetching data
        try:
            data = next(train_itr)
        except StopIteration:
            train_itr = iter(train_loader)
            data = next(train_itr)
        gt_img, camera, gt_mask = data

        gt_img = gt_img[0].cuda()  # (H, W, 3)
        camera = camera[0].cuda()  # PerspectiveCameras()
        if gt_mask is not None:
            gt_mask = gt_mask[0].cuda()  # (H, W, 1)

        # Render scene using gaussian splatting
        pred_img, pred_depth, pred_mask = scene.render(
            camera,
            per_splat=args.gaussians_per_splat,
            img_size=train_dataset.img_size,
            bg_colour=(0.0, 0.0, 0.0),
        )

        # Compute MSE loss. TODO: Should I normalize the image?
        loss = mse_loss(pred_img, gt_img)
        # loss = criterion(pred_img, gt_img)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f}")

        # Detach the tensors
        gt_img.detach_().cpu()
        # camera.detach_().cpu()
        if gt_mask is not None:
            gt_mask.detach_().cpu()
        pred_img.detach_().cpu()

        # Delete variables to free memory
        del gt_img, camera, gt_mask, pred_img  # , loss
        gc.collect()

        # Empty the CUDA cache
        torch.cuda.empty_cache()

        if itr % args.viz_freq == 0:
            viz_frame = visualize_renders(
                scene, gt_viz_img, viz_cameras, train_dataset.img_size
            )
            viz_frames.append(viz_frame)

    print("[*] Training Completed.")

    # Saving training progess GIF
    imageio.mimwrite(viz_gif_path_1, viz_frames, loop=0, duration=(1 / 10.0) * 1000)

    # Creating renderings of the training views after training is completed.
    frames = []
    viz_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=CowDataset.collate_fn,
    )
    for viz_data in tqdm(viz_loader, desc="Creating Visualization"):
        gt_img, camera, gt_mask = viz_data
        gt_img = gt_img[0].cuda()
        camera = camera[0].cuda()
        if gt_mask is not None:
            gt_mask = gt_mask[0].cuda()

        with torch.no_grad():
            # Rendering scene using gaussian splatting
            pred_img, *_ = scene.render(
                camera=camera,
                per_splat=args.gaussians_per_splat,
                img_size=train_dataset.img_size,
                bg_colour=(0.0, 0.0, 0.0),
            )

        pred_npy = pred_img.detach().cpu().numpy()
        pred_npy = (np.clip(pred_npy, 0.0, 1.0) * 255.0).astype(np.uint8)
        frames.append(pred_npy)

    # Saving renderings
    imageio.mimwrite(viz_gif_path_2, frames, loop=0, duration=(1 / 10.0) * 1000)

    # Running evaluation using the test dataset
    psnr_vals, ssim_vals = [], []
    for test_data in tqdm(test_loader, desc="Running Evaluation"):
        gt_img, camera, gt_mask = test_data
        gt_img = gt_img[0].cuda()
        camera = camera[0].cuda()
        if gt_mask is not None:
            gt_mask = gt_mask[0].cuda()

        with torch.no_grad():
            # Rendering scene using gaussian splatting
            pred_img, *_ = scene.render(
                camera=camera,
                per_splat=args.gaussians_per_splat,
                img_size=test_dataset.img_size,
                bg_colour=(0.0, 0.0, 0.0),
            )

            gt_npy = gt_img.detach().cpu().numpy()
            pred_npy = pred_img.detach().cpu().numpy()
            psnr = peak_signal_noise_ratio(gt_npy, pred_npy)
            ssim = structural_similarity(
                gt_npy, pred_npy, channel_axis=-1, data_range=1.0
            )

            psnr_vals.append(psnr)
            ssim_vals.append(ssim)

    mean_psnr = np.mean(psnr_vals)
    mean_ssim = np.mean(ssim_vals)
    print(f"[*] Evaluation --- Mean PSNR: {mean_psnr:.3f}")
    print(f"[*] Evaluation --- Mean SSIM: {mean_ssim:.3f}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path",
        default="./output",
        type=str,
        help="Path to the directory where output should be saved to.",
    )
    parser.add_argument(
        "--data_path",
        default="./data/cow_dataset",
        type=str,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--gaussians_per_splat",
        default=-1,
        type=int,
        help=(
            "Number of gaussians to splat in one function call. If set to -1, "
            "then all gaussians in the scene are splat in a single function call. "
            "If set to any other positive interger, then it determines the number of "
            "gaussians to splat per function call (the last function call might splat "
            "lesser number of gaussians). In general, the algorithm can run faster "
            "if more gaussians are splat per function call, but at the cost of higher GPU "
            "memory consumption."
        ),
    )
    parser.add_argument(
        "--num_itrs",
        default=50,  # 2000
        type=int,
        help="Number of iterations to train the model.",
    )
    parser.add_argument(
        "--viz_freq",
        default=5,  # 20
        type=int,
        help="Frequency with which visualization should be performed.",
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    parser.add_argument("--isotropic", default=None, help="3D Gaussian is isotropic.")
    parser.add_argument(
        "--init_type",
        required=True,
        type=str,
        help="Data type to be initialized.",
    )
    parser.add_argument("--dataset-name", type=str)
    args = parser.parse_args()
    return args


# python3 train.py --init_type points --> cow data
# python3 train.py --init_type points --isotropic False --> cow data anisotropic
# python3 train.py --init_type gaussians --data_path ./data/sledge.ply --> sledge data
# python3 train.py --init_type gaussians --data_path ./data/materials --dataset-name "materials"
if __name__ == "__main__":
    args = get_args()
    run_training(args)
