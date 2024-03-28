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
    """Implement Figure 7 from [Mildenhall et al. 2020], an MLP with a skip connection."""

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
        # NOTE: Python is pass-by-reference and torch.Tensor is mutable.
        # When passing the same Tensor 'Original' as both `x` and `skip_pos` to this
        # function, `x` and `skip_pos` will have the same `id` as 'Original'.
        # Interestingly, when using `ReLU(inplace=True)`, `x` will not overwrite the
        # original 'Original' Tensor and `skip_pos` Tensor.
        # Python will create new object for `x` when doing `x = layer(x)` such that the
        # original 'Original' Tensor and `skip_pos` Tensor will not be effected.
        # tmp_id = id(skip_pos)

        # Ok, scratch what I wrote above. It seems `x` needs to be reassigned to another
        # object so the whole program will work. Python is weird.
        copied_x = x
        for i, layer in enumerate(self._mlp):
            if i in self._input_skips:
                copied_x = torch.cat((copied_x, skip_pos), dim=-1)

            copied_x = layer(copied_x)

        # assert tmp_id == id(skip_pos)
        return copied_x


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

    def forward(self, ray_bundle):
        # ray_bundle: (num_rays, )
        pos = ray_bundle.sample_points  # (num_rays, n_pts_per_ray, 3)
        dir = ray_bundle.directions  # (num_rays, 3)
        position_encoding = self.harmonic_embedding_xyz(pos)  # (1024, 128, 36)
        direction_encoding = self.harmonic_embedding_dir(dir)  # (1024, 12)
        # tmp = position_encoding.clone()
        xyz_out = self.xyz_out_layer(
            position_encoding, position_encoding
        )  # (1024, 160, 128)
        # assert torch.equal(position_encoding, tmp)
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
