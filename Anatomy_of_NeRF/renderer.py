import torch


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = (
            cfg.white_background if "white_background" in cfg else False
        )

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

    def _normalize_compute_weights(
        self, deltas, rays_density: torch.Tensor, eps: float = 1e-10
    ):
        # FIXME: this doesn't work.
        # deltas (𝛿): (num_rays, num_samples, 1)
        # rays_density (σ): (num_rays, num_samples, 1)
        # In theory, deltas and rays_density shall apply to the following rules. In case
        # of rays_density is 0, I add `eps` to it. If tmp_multiply is 0, then it's
        # possible the output weight w_i is 0, which is bad.
        # assert torch.all(deltas > 0)
        # assert torch.all(rays_density >= 0)
        tmp_multiply = deltas * (rays_density)  # (num_rays, num_samples, 1)

        # Calculate the transmittance `T_i` along the ray for each sample point
        T_i = torch.exp(-torch.cumsum(tmp_multiply, dim=1)).to(rays_density.device)

        # Calculate weights `w_i` for each sample
        w_i = T_i * (1 - torch.exp(-tmp_multiply))  # (num_rays, num_samples, 1)
        w_i /= torch.sum(w_i, dim=1, keepdim=True)
        return w_i

    def _inverse_transform_sample(self, ray_bundle, deltas, normalized_weights):
        """Resample points based on normalized weights along the ray.
        deltas: (num_rays, num_points, 1)
        normalized_weights: (num_rays, num_points, 1)
        """
        # FIXME: this doesn't work.
        num_rays, num_samples = deltas.shape[:2]

        # Calculate the cumulative distribution function (CDF) from normalized weights
        cdf = torch.cumsum(normalized_weights, dim=1)

        # Uniform sampling in the cumulative distribution space
        uniform_samples = torch.rand(num_rays, num_samples, 1, device=deltas.device)

        # Inverse transform sampling to find the indices in the CDF
        indices = torch.searchsorted(cdf, uniform_samples, right=True)

        # Ensure indices are within bounds
        indices = torch.clamp(indices, 0, num_samples - 1)

        # Gather new sample points based on indices
        gathered_points = torch.gather(
            ray_bundle.sample_points, 1, indices.expand(-1, -1, 3)
        )
        gathered_lengths = torch.gather(ray_bundle.sample_lengths, 1, indices)

        # Replace old samples with new samples in the ray bundle
        new_ray_bundle = ray_bundle._replace(
            sample_points=gathered_points, sample_lengths=gathered_lengths
        )

        return new_ray_bundle

    def _aggregate(self, weights: torch.Tensor, rays_feature: torch.Tensor):
        # Aggregate (weighted sum of) features using weights

        # weights: (num_rays, num_samples, 1)
        # rays_feature: (num_rays, num_samples, num_channel)
        num_rays, num_samples, num_channel = rays_feature.shape

        feature = torch.sum(weights * rays_feature, dim=1)
        assert feature.shape == (num_rays, num_channel)
        return feature

    def forward(self, sampler, implicit_fn, ray_bundle):
        num_rays = ray_bundle.shape[0]

        # Process the chunks of rays.
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

            # # After calculating coarse, we now calculate fine
            # normalized_weights = self._normalize_compute_weights(
            #     deltas.view(-1, num_points, 1), coarse_density.view(-1, num_points, 1)
            # )  # (num_rays, num_points, 1)

            # # Hierarchical sampling based on normalized weights
            # fine_ray_bundle = self._inverse_transform_sample(
            #     curr_ray_bundle, deltas, normalized_weights
            # )
            # # Call implicit function with resampled points (fine sampling)
            # fine_output = implicit_fn(fine_ray_bundle)  # NeRF MLP
            # fine_density = fine_output["density"]  # (num_rays * num_points, 1)
            # fine_feature = fine_output["feature"]  # (num_rays * num_points, 3)
            # fine_depth_values = fine_ray_bundle.sample_lengths[
            #     ..., 0
            # ]  # (num_rays, num_points)
            # fine_deltas = torch.cat(
            #     (
            #         fine_depth_values[..., 1:] - fine_depth_values[..., :-1],
            #         1e10 * torch.ones_like(fine_depth_values[..., :1]),
            #     ),
            #     dim=-1,
            # )[..., None]
            # fine_weights = self._compute_weights(
            #     fine_deltas.view(-1, num_points, 1),
            #     fine_density.view(-1, num_points, 1),
            # )
            # fine_feature = self._aggregate(
            #     fine_weights, fine_feature.view(-1, num_points, 3)
            # )
            # fine_depth = self._aggregate(fine_weights, fine_depth_values.unsqueeze(-1))

            # Return
            cur_out = {
                "feature": coarse_feature,
                "depth": coarse_depth,
                # "fine_feature": fine_feature,
                # "fine_depth": fine_depth,
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
