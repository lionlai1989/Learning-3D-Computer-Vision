import torch


class VolumeRenderer(torch.nn.Module):
    """
    A volume renderer based on the principles outlined in [Mildenhall et al. 2020],
    designed to integrate color and density information along rays to produce
    images from 3D volumetric data.

    Parameters:
        chunk_size (int): The size of data chunks processed at a time.
        white_background (bool): Not used currently.
    """

    def __init__(self, chunk_size, white_background, **kwargs):
        super().__init__()

        self._chunk_size = chunk_size
        self._white_background = white_background

    def _compute_weights(self, deltas, rays_density: torch.Tensor, eps: float = 1e-10):
        # deltas (𝛿): (num_rays, num_samples, 1)
        # rays_density (σ): (num_rays, num_samples, 1)
        # In theory, deltas and rays_density shall apply to the following rules. In case
        # of rays_density is 0, I add `eps` to it. If tmp_multiply is 0, then it's
        # possible the output weight w_i is 0, which is bad.
        assert torch.all(deltas > 0)
        assert torch.all(rays_density >= 0)
        tmp_multiply = deltas * (rays_density)  # (num_rays, num_samples, 1)

        # Calculate the transmittance `T_i` along the ray for each sample point
        T_i = torch.exp(-torch.cumsum(tmp_multiply, dim=1)).to(rays_density.device)

        # Calculate weights `w_i` for each sample
        w_i = T_i * (1 - torch.exp(-tmp_multiply))  # FIXME: where is eps?
        return w_i

    def _aggregate(self, weights: torch.Tensor, rays_feature: torch.Tensor):
        # Aggregate weighted sum of features using weights
        num_rays, num_samples, num_channel = rays_feature.shape
        assert weights.shape == (num_rays, num_samples, 1)

        feature = torch.sum(weights * rays_feature, dim=1)
        assert feature.shape == (num_rays, num_channel)
        return feature

    def forward(self, sampler, implicit_fn, ray_bundle):
        num_rays = ray_bundle.shape[0]

        chunk_outputs = []

        for chunk_start in range(0, num_rays, self._chunk_size):
            curr_ray_bundle = ray_bundle[chunk_start : chunk_start + self._chunk_size]

            # Sample points along the ray
            curr_ray_bundle = sampler(curr_ray_bundle)
            num_points = curr_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            coarse_output = implicit_fn(curr_ray_bundle)  # NeRF MLP
            coarse_density = coarse_output["density"]  # (num_rays * num_points, 1)
            coarse_feature = coarse_output["feature"]  # (num_rays * num_points, 3)

            # Compute length of each ray segment
            depth_values = curr_ray_bundle.sample_lengths[
                ..., 0
            ]  # (num_rays, num_points)
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute coarse aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, num_points, 1), coarse_density.view(-1, num_points, 1)
            )  # (num_rays, num_points, 1)
            # Render (color) features using weights
            coarse_feature = self._aggregate(
                weights, coarse_feature.view(-1, num_points, 3)
            )
            # Render depth map
            coarse_depth = self._aggregate(weights, depth_values.unsqueeze(-1))

            cur_out = {
                "feature": coarse_feature,
                "depth": coarse_depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat([chunk_out[k] for chunk_out in chunk_outputs], dim=0)
            for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    "volume": VolumeRenderer,
}
