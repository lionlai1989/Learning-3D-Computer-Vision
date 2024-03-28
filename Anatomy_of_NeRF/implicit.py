import torch
import torch.nn.functional as F


class HarmonicEmbedding(torch.nn.Module):
    """Implement position encoding in NeRF."""

    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
    ) -> None:
        super().__init__()

        frequencies = 2.0 ** torch.arange(n_harmonic_functions, dtype=torch.float32)
        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.output_dim = n_harmonic_functions * 2 * in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class MLPWithInputSkips(torch.nn.Module):
    """Implement Figure 7 from [Mildenhall et al. 2020], an MLP with a skip connection."""

    def __init__(
        self, n_layers, input_dim, output_dim, skip_dim, hidden_dim, input_skips
    ):
        super().__init__()
        self._input_skips = set(input_skips)

        layers = []
        for i in range(n_layers):
            # The first and last layer can also be skip layer.
            dimin = input_dim if i == 0 else hidden_dim  # first layer
            dimout = output_dim if i == n_layers - 1 else hidden_dim  # last layer
            if i in self._input_skips:
                dimin += skip_dim

            linear = torch.nn.Linear(dimin, dimout)

            if i < n_layers - 1:
                layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(inplace=True)))
            elif i == n_layers - 1:  # Last layer has no activation
                layers.append(torch.nn.Sequential(linear))

        self._mlp = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, skip_pos: torch.Tensor) -> torch.Tensor:
        # NOTE: Python is pass-by-reference, torch.Tensor is mutable and `ReLU(inplace=True)`.
        # Does the following code have any UNEXPEXTED side effect when `x is skip_pos`?
        # tmp_id = id(skip_pos)
        for i, layer in enumerate(self._mlp):
            if i in self._input_skips:
                x = torch.cat((x, skip_pos), dim=-1)
            x = layer(x)
        # assert tmp_id == id(skip_pos)
        return x


class NeuralRadianceField(torch.nn.Module):
    """Implement NeRF."""

    def __init__(self, cfg):
        super().__init__()
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = (
            self.harmonic_embedding_xyz.output_dim
        )  # 3 * n_harmonic_functions_xyz * 2
        embedding_dim_dir = (
            self.harmonic_embedding_dir.output_dim
        )  # 3 * n_harmonic_functions_dir * 2

        self.xyz_out_layer = MLPWithInputSkips(
            n_layers=cfg.n_layers_xyz,
            input_dim=embedding_dim_xyz,
            output_dim=cfg.n_hidden_neurons_xyz,
            skip_dim=embedding_dim_xyz,
            hidden_dim=cfg.n_hidden_neurons_xyz,
            input_skips=cfg.append_xyz,
        )
        self.density_out_layer = torch.nn.Sequential(
            torch.nn.Linear(cfg.n_hidden_neurons_xyz, 1),
            torch.nn.ReLU(inplace=True),  # ensure density being nonnegative
        )
        self.feature_vector = torch.nn.Sequential(
            torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz),
            torch.nn.ReLU(inplace=True),
        )

        self.rgb_out_layer = torch.nn.Sequential(
            torch.nn.Linear(
                embedding_dim_dir + cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_dir
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(cfg.n_hidden_neurons_dir, 3),
            torch.nn.Sigmoid(),  # (r, g, b) in [0, 1]
        )

    def forward(self, ray_bundle):  # ray_bundle: (num_rays, )
        pos = ray_bundle.sample_points  # (num_rays, n_pts_per_ray, 3)
        dir = ray_bundle.directions  # (num_rays, 3)
        position_encoding = self.harmonic_embedding_xyz(pos)
        direction_encoding = self.harmonic_embedding_dir(dir)
        # tmp = position_encoding.clone()
        xyz_out = self.xyz_out_layer(position_encoding, position_encoding)
        # assert torch.equal(position_encoding, tmp)
        density = self.density_out_layer(xyz_out)
        feature = self.feature_vector(xyz_out)

        expanded_direction_encoding = direction_encoding.unsqueeze(1).repeat(
            1, feature.shape[1], 1
        )  # (num_rays, 24) --> (num_rays, 1, 24) --> (num_rays, n_pts_per_ray, 24)

        # Concatenate feature and expanded_direction_encoding
        rgb = self.rgb_out_layer(
            torch.cat([feature, expanded_direction_encoding], dim=-1)
        )

        return {"density": density, "feature": rgb}


implicit_dict = {
    "nerf": NeuralRadianceField,
}
