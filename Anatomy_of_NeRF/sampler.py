import torch


class StratifiedRaysampler(torch.nn.Module):
    """
    Implements stratified sampling along rays in a 3D space, ensuring uniform distribution of
    sample points between minimum and maximum depths.

    Parameters:
        n_pts_per_ray (int): Number of points to sample along each ray.
        min_depth (float): Minimum depth from which to start sampling along the ray.
        max_depth (float): Maximum depth at which to end sampling along the ray.
    """

    def __init__(self, cfg):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(self, ray_bundle):
        # Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(
            start=self.min_depth, end=self.max_depth, steps=self.n_pts_per_ray
        ).to(ray_bundle.origins.device)
        z_vals = z_vals.view(
            1, -1, 1
        )  # Convert `torch.Size([64])` to `torch.Size([1, 64, 1])`

        # Sample points from z values
        directions_view = ray_bundle.directions.view(
            -1, 1, 3
        )  # `(num_rays, 3)` --> `(num_rays, 1, 3)`
        origins_view = ray_bundle.origins.view(
            -1, 1, 3
        )  # `(num_rays, 3)` --> `(num_rays, 1, 3)`
        # (num_rays, n_pts_per_ray, 3) = (1, n_pts_per_ray, 1) * (num_rays, 1, 3) + (num_rays, 1, 3)
        sample_points = z_vals * directions_view + origins_view

        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {"stratified": StratifiedRaysampler}
