import os
from pathlib import Path

import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig
from torchinfo import summary

from data_utils import (
    create_cameras_fern,
    create_cameras_lego,
    create_cameras_materials,
    dataset_from_config,
)
from dataset import get_nerf_datasets, trivial_collate
from implicit import implicit_dict
from ray_utils import (
    get_pixels_from_image,
    get_random_pixels_from_image,
    get_rays_from_pixels,
    sample_images_at_xy,
)
from renderer import renderer_dict
from sampler import sampler_dict


class Model(torch.nn.Module):
    """
    A PyTorch module for rendering scenes defined by implicit volumes.

    This class encapsulates the concept of an implicit volume that defines a scene,
    along with a sampling scheme to generate sample points along rays. It integrates
    these components with a renderer capable of producing images from the defined
    implicit volume using the provided sampling scheme.
    """

    def __init__(self, cfg):
        super().__init__()

        # Get implicit function from config
        self.implicit_fn = implicit_dict[cfg.implicit_function.type](
            cfg.implicit_function
        )

        # Point sampling (raymarching) scheme
        self.sampler = sampler_dict[cfg.sampler.type](**cfg.sampler)

        # Initialize volume renderer
        self.renderer = renderer_dict[cfg.renderer.type](**cfg.renderer)

    def forward(self, ray_bundle):
        # Call renderer with
        #  a) Implicit volume
        #  b) Sampling routine

        return self.renderer(self.sampler, self.implicit_fn, ray_bundle)


def render_images(model, cameras, image_size, save=False, file_prefix=""):
    all_images = []
    device = list(model.parameters())[0].device

    print(f"Rendering model with {len(cameras)} cameras.")
    for cam_idx, camera in enumerate(cameras):

        torch.cuda.empty_cache()
        camera = camera.to(device)
        xy_grid = get_pixels_from_image(image_size, camera)  # (65536, 2)
        ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera)

        out = model(ray_bundle)

        image = np.array(
            out["feature"].view(image_size[1], image_size[0], 3).detach().cpu()
        )  # Get rendered colors
        all_images.append(image)

        if save:
            print("image:", image.shape, image.dtype)
            plt.imsave(f"{file_prefix}_{cam_idx}.png", image)

    return all_images


def display_training_data(cfg):
    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name=cfg.data.dataset_name,
        image_size=[cfg.data.image_size[1], cfg.data.image_size[0]],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=trivial_collate,
    )
    print(f"Rendering {len(train_dataloader)} views on {cfg.data.dataset_name} data.")

    all_images = []
    all_cameras = []
    for index, data in enumerate(train_dataloader):
        # data: [{"image": tensor, "camera": PerspectiveCameras(), "camera_idx": int}]
        image = data[0]["image"]  # Tensor, (400, 400, 3], float32, 0., 1.
        camera = data[0]["camera"]  # PerspectiveCameras()
        all_images.append(image)
        all_cameras.append(camera)

    output_path = (
        Path("./output/rendering_training_data") / f"{cfg.data.dataset_name}.gif"
    )
    imageio.mimsave(
        output_path, [np.uint8(im * 255) for im in all_images], fps=2, loop=0
    )
    # TODO: Plot cameras' position and viewing direction in 3D.


def create_model(cfg):
    def find_checkpoint(checkpoint_dir):
        """Find the checkpoint with largest epoch number."""
        files = sorted(
            Path(checkpoint_dir).glob(f"{cfg.data.dataset_name}_epoch_*.pth")
        )
        if len(files) == 0:
            return None
        else:
            return files[-1]

    # Create model
    model = Model(cfg)
    model.cuda()
    model.train()

    # Load checkpoints
    optimizer_state_dict = None
    start_epoch = 0

    if cfg.training.checkpoint_path is not None:
        checkpoint_dir = os.path.join(
            hydra.utils.get_original_cwd(), cfg.training.checkpoint_path
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume training if requested.
        checkpoint_file = find_checkpoint(checkpoint_dir)
        if cfg.training.resume and checkpoint_file is not None:
            print(f"Resuming from checkpoint {checkpoint_file}.")
            loaded_data = torch.load(checkpoint_file)
            model.load_state_dict(loaded_data["model"])
            start_epoch = loaded_data["epoch"]

            print(f"   => resuming from epoch {start_epoch}.")
            optimizer_state_dict = loaded_data["optimizer"]
    else:
        checkpoint_dir = None

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
    )

    # Load the optimizer state dict in case we are resuming.
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        optimizer.last_epoch = start_epoch - 1

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    def lr_lambda(epoch):
        return cfg.training.lr_scheduler_gamma ** (
            epoch / cfg.training.lr_scheduler_step_size
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1
    )

    if checkpoint_file is not None:
        start_epoch += 1
    return model, optimizer, lr_scheduler, start_epoch, checkpoint_dir


def train_nerf(cfg):
    # Create model
    model, optimizer, lr_scheduler, start_epoch, checkpoint_dir = create_model(cfg)
    print(f"The current learning rate: {lr_scheduler.get_last_lr()}")
    print(f"NeRF architecture: {model.implicit_fn}")

    # Load the training/validation data.
    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name=cfg.data.dataset_name,
        image_size=[cfg.data.image_size[1], cfg.data.image_size[0]],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=trivial_collate,
    )

    coarse_criteria = torch.nn.MSELoss()
    fine_criteria = torch.nn.MSELoss()

    # Run the main training loop.
    for epoch in range(start_epoch, cfg.training.num_epochs + 1):
        t_range = tqdm.tqdm(enumerate(train_dataloader))

        for iteration, batch in t_range:
            image, camera, camera_idx = batch[0].values()
            image = image.cuda().unsqueeze(0)
            camera = camera.cuda()

            # Sample rays
            xy_grid = get_random_pixels_from_image(
                cfg.training.batch_size, cfg.data.image_size, camera
            )
            rgb_gt = sample_images_at_xy(image, xy_grid)

            coarse_ray_bundle = get_rays_from_pixels(
                xy_grid, cfg.data.image_size, camera
            )
            # Run model forward
            out = model(coarse_ray_bundle)

            loss = coarse_criteria(out["feature"], rgb_gt)
            # TODO: Finish hierarchical sampling. loss = coarse_criteria(out["feature"], rgb_gt) + fine_criteria(out["fine_feature"], rgb_gt)

            # Take the training step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_range.set_description(f"Epoch: {epoch:04d}, Loss: {loss:.06f}")
            t_range.refresh()

        # Adjust the learning rate.
        lr_scheduler.step()

        if (
            epoch % cfg.training.checkpoint_interval == 0
            and cfg.training.checkpoint_path is not None
            and epoch >= 0
        ):
            checkpoint_outfile = (
                Path(checkpoint_dir) / f"{cfg.data.dataset_name}_epoch_{epoch:05}.pth"
            )
            print(f"Storing checkpoint {checkpoint_outfile}.")

            data_to_store = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

            torch.save(data_to_store, checkpoint_outfile)

        # Render
        if epoch % cfg.training.render_interval == 0 and epoch >= 0:
            with torch.no_grad():
                if cfg.data.dataset_name == "fern":
                    cameras = create_cameras_fern(
                        0.1, n_poses=20, up=(0.0, 1.0, 0.0), focal_length=1.2
                    )
                elif cfg.data.dataset_name == "lego":
                    cameras = create_cameras_lego(
                        4.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0
                    )
                elif cfg.data.dataset_name == "materials":
                    cameras = create_cameras_materials(
                        2.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0
                    )
                test_images = render_images(
                    model,
                    cameras,
                    cfg.data.image_size,
                    file_prefix="nerf",
                )
                imageio.mimsave(
                    f"output/{cfg.data.dataset_name}_epoch_{epoch:05}.gif",
                    [np.uint8(im * 255) for im in test_images],
                    fps=5,
                    loop=0,  # Make gif loop indefinitely.
                )

        torch.cuda.empty_cache()


@hydra.main(config_path="./configs", config_name="box", version_base=None)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    if cfg.type == "display":
        display_training_data(cfg)
    elif cfg.type == "train_nerf":
        train_nerf(cfg)


# python3 main.py --config-name=train_fern_lowres
# python3 main.py --config-name=train_lego_lowres
# python3 main.py --config-name=train_materials_lowres
if __name__ == "__main__":
    main()
