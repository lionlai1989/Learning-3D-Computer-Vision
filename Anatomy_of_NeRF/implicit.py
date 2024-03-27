import torch
import torch.nn.functional as F


class HarmonicEmbedding(torch.nn.Module):
    """Implement position encoding in NeRF."""

    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class MLPWithInputSkips(torch.nn.Module):
    """Implement Fig. 7 where MLP with an input skip connection."""
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for i, layeri in enumerate(range(n_layers)):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)

            if i < n_layers - 1:
                layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
            else:  # `i == n_layers - 1`: Last layer has no activation
                layers.append(torch.nn.Sequential(linear))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, viewing_direction: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, viewing_direction), dim=-1)

            y = layer(y)

        return y


class NeuralRadianceField(torch.nn.Module):
    """Implement NeRF."""

    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.harmonic_embedding_xyz = HarmonicEmbedding(
            3, cfg.n_harmonic_functions_xyz, include_input=False
        )
        self.harmonic_embedding_dir = HarmonicEmbedding(
            3, cfg.n_harmonic_functions_dir, include_input=False
        )

        embedding_dim_xyz = (
            self.harmonic_embedding_xyz.output_dim
        )  # 3 * n_harmonic_functions_xyz * 2
        embedding_dim_dir = (
            self.harmonic_embedding_dir.output_dim
        )  # 3 * n_harmonic_functions_dir * 2

        self.xyz_out_layer = MLPWithInputSkips(
            n_layers=cfg.n_layers_xyz,
            input_dim=embedding_dim_xyz,
            output_dim=128,
            skip_dim=embedding_dim_xyz,
            hidden_dim=cfg.n_hidden_neurons_xyz,
            input_skips=cfg.append_xyz,
        )
        self.density_out_layer = torch.nn.Sequential(
            torch.nn.Linear(cfg.n_hidden_neurons_xyz, 1),
            torch.nn.ReLU(),  # ensure density being nonnegative
        )
        self.feature_vector = torch.nn.Sequential(
            torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz),
            torch.nn.ReLU(),
        )

        self.rgb_out_layer = torch.nn.Sequential(
            torch.nn.Linear(
                embedding_dim_dir + cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_dir
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg.n_hidden_neurons_dir, 3),
            torch.nn.Sigmoid(),  # (r, g, b) in [0, 1]
        )

    def forward(self, ray_bundle):
        # ray_bundle: (num_rays, )
        pos = ray_bundle.sample_points  # (num_rays, n_pts_per_ray, 3)
        dir = ray_bundle.directions  # (num_rays, 3)
        position_encoding = self.harmonic_embedding_xyz(pos)  # (1024, 128, 36)
        direction_encoding = self.harmonic_embedding_dir(dir)  # (1024, 12)

        xyz_out = self.xyz_out_layer(
            position_encoding, position_encoding
        )  # (1024, 160, 128)
        density = self.density_out_layer(xyz_out)  # (1024, 160, 1)
        feature = self.feature_vector(xyz_out)  # (1024, 160, 128)

        expanded_direction_encoding = direction_encoding.unsqueeze(1).repeat(
            1, feature.shape[1], 1
        )  # (1024, 12) --> (1024, 1, 12) --> (1024, 160, 12)

        # Concatenate feature and expanded_direction_encoding
        rgb = self.rgb_out_layer(
            torch.cat([feature, expanded_direction_encoding], dim=-1)
        )

        return {"density": density, "feature": rgb}


implicit_dict = {
    "nerf": NeuralRadianceField,
}
