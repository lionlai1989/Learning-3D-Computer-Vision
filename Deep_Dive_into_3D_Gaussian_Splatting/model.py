import math
from typing import Optional, Tuple

import numpy as np
import torch
from data_utils import load_gaussians_from_ply, colours_from_spherical_harmonics
from pytorch3d.ops.knn import knn_points
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.transforms import quaternion_to_matrix


class Gaussians:
    """Implementation of the 3D Gaussians described in the paper,
    "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (3DGS).

    This class provides methods and attributes to handle 3D Gaussian splats, which
    are used for real-time rendering of radiance fields. The 3D Gaussian splatting
    technique models the scene as a collection of anisotropic Gaussian functions.

    Link:
        https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf

    """

    def __init__(
        self,
        init_type: str,
        device: str,
        load_path: Optional[str] = None,
        num_points: Optional[int] = None,
        isotropic: Optional[bool] = None,
    ):
        if device not in ("cpu", "cuda"):
            raise ValueError(f"Unsupported device: {device}")
        else:
            self.device = device

        if init_type == "gaussians":
            print("type: ", isotropic, type(isotropic))
            if isotropic is not None:
                raise ValueError(
                    (
                        "Isotropy/Anisotropy will be determined from pre-trained gaussians. "
                        "Please set isotropic to None."
                    )
                )
            if load_path is None:
                raise ValueError

            data, is_isotropic = self._load_gaussians(load_path)
            self.is_isotropic = is_isotropic

        elif init_type == "points":
            if isotropic is not None and not isinstance(isotropic, bool):
                raise TypeError("isotropic must be either None or True or False.")
            if load_path is None:
                raise ValueError

            if isotropic is None:
                self.is_isotropic = False
            else:
                self.is_isotropic = isotropic

            data = self._load_points(load_path)

        elif init_type == "random":
            if isotropic is not None and not isinstance(isotropic, bool):
                raise TypeError("isotropic must be either None or True or False.")
            if num_points is None:
                raise ValueError

            if isotropic is None:
                self.is_isotropic = False
            else:
                self.is_isotropic = isotropic

            data = self._load_random(num_points)

        else:
            raise ValueError(f"Invalid init_type: {init_type}")

        self.pre_act_quats = data["pre_act_quats"]
        self.means = data["means"]
        self.pre_act_scales = data["pre_act_scales"]
        self.colours = data["colours"]
        self.pre_act_opacities = data["pre_act_opacities"]

        # [Q 1.3.1] NOTE: Uncomment spherical harmonics code for question 1.3.1
        if data.get("spherical_harmonics") is not None:
            self.spherical_harmonics = data["spherical_harmonics"]
        else:
            print("Not using spherical harmonics.")

        if self.device == "cuda":
            self.to_cuda()
        else:
            print("Warning: it's not using GPU.")

    def __len__(self):
        return len(self.means)

    def _load_gaussians(self, ply_path: str):
        data = dict()
        print("ply_path: ", ply_path)
        ply_gaussians = load_gaussians_from_ply(ply_path)

        data["means"] = torch.tensor(ply_gaussians["xyz"])
        data["pre_act_quats"] = torch.tensor(ply_gaussians["rot"])
        data["pre_act_scales"] = torch.tensor(ply_gaussians["scale"])
        data["pre_act_opacities"] = torch.tensor(ply_gaussians["opacity"]).squeeze()
        data["colours"] = torch.tensor(ply_gaussians["dc_colours"])

        # [Q 1.3.1] NOTE: Uncomment spherical harmonics code for question 1.3.1
        data["spherical_harmonics"] = torch.tensor(ply_gaussians["sh"])

        if data["pre_act_scales"].shape[1] != 3:
            raise NotImplementedError("Currently does not support isotropic")

        is_isotropic = False

        return data, is_isotropic

    def _load_points(self, path: str):
        data = dict()
        means = np.load(path)

        # Initializing means using the provided point cloud
        data["means"] = torch.tensor(means.astype(np.float32))  # (N, 3)

        # Initializing opacities such that all when sigmoid is applied to pre_act_opacities,
        # we will have a opacity value close to (but less than) 1.0
        data["pre_act_opacities"] = 8.0 * torch.ones(
            (len(means),), dtype=torch.float32
        )  # (N,)

        # Initializing colors randomly
        data["colours"] = torch.rand((len(means), 3), dtype=torch.float32)  # (N, 3)

        # Initializing quaternions to be the identity quaternion
        quats = torch.zeros((len(means), 4), dtype=torch.float32)  # (N, 4)
        quats[:, 0] = 1.0
        data["pre_act_quats"] = quats  # (N, 4)

        # Initializing scales using the mean distance of each point to its 50 nearest points
        dists, _, _ = knn_points(
            data["means"].unsqueeze(0), data["means"].unsqueeze(0), K=50
        )
        data["pre_act_scales"] = torch.log(torch.mean(dists[0], dim=1)).unsqueeze(
            1
        )  # (N, 1)

        if not self.is_isotropic:
            # If anisotropic, there are three individual directions.
            data["pre_act_scales"] = data["pre_act_scales"].repeat(1, 3)  # (N, 3)

        return data

    def _load_random(self, num_points: int):
        data = dict()

        # Initializing means randomly
        data["means"] = torch.randn((num_points, 3)).to(torch.float32) * 0.2  # (N, 3)

        # Initializing opacities such that all when sigmoid is applied to pre_act_opacities,
        # we will have a opacity value close to (but less than) 1.0
        data["pre_act_opacities"] = 8.0 * torch.ones(
            (num_points,), dtype=torch.float32
        )  # (N,)

        # Initializing colors randomly
        data["colours"] = torch.rand((num_points, 3), dtype=torch.float32)  # (N, 3)

        # Initializing quaternions to be the identity quaternion
        quats = torch.zeros((num_points, 4), dtype=torch.float32)  # (N, 4)
        quats[:, 0] = 1.0
        data["pre_act_quats"] = quats  # (N, 4)

        # Initializing scales randomly
        data["pre_act_scales"] = torch.log(
            (torch.rand((num_points, 1), dtype=torch.float32) + 1e-6) * 0.01
        )

        if not self.is_isotropic:
            data["pre_act_scales"] = data["pre_act_scales"].repeat(1, 3)  # (N, 3)

        return data

    def _compute_jacobian(
        self, means_3D: torch.Tensor, camera: PerspectiveCameras, img_size: Tuple
    ):
        """TODO: Find out the source for the following code."""

        if camera.in_ndc():
            raise RuntimeError

        fx, fy = camera.focal_length.flatten()
        width, height = img_size

        half_tan_fov_x = 0.5 * width / fx
        half_tan_fov_y = 0.5 * height / fy

        view_transform = camera.get_world_to_view_transform()
        means_view_space = view_transform.transform_points(means_3D)  # (N, 3, 3)

        tx = means_view_space[:, 0]
        ty = means_view_space[:, 1]
        tz = means_view_space[:, 2]  # (N, 3)
        tz2 = tz * tz

        lim_x = 1.3 * half_tan_fov_x
        lim_y = 1.3 * half_tan_fov_y

        tx = torch.clamp(tx / tz, -lim_x, lim_x) * tz
        ty = torch.clamp(ty / tz, -lim_y, lim_y) * tz

        J = torch.zeros((len(tx), 2, 3))  # (N, 2, 3)
        J = J.to(self.device)

        J[:, 0, 0] = fx / tz
        J[:, 1, 1] = fy / tz
        J[:, 0, 2] = -(fx * tx) / tz2
        J[:, 1, 2] = -(fy * ty) / tz2

        return J  # (N, 2, 3)

    def check_if_trainable(self):
        attrs = ["means", "pre_act_scales", "colours", "pre_act_opacities"]
        if not self.is_isotropic:
            attrs += ["pre_act_quats"]

        for attr in attrs:
            param = getattr(self, attr)
            if not getattr(param, "requires_grad", False):
                raise Exception(
                    f"Please use function make_trainable to make parameters '{attr}' trainable"
                )

        if self.is_isotropic and self.pre_act_quats.requires_grad:
            raise RuntimeError(
                "You do not need to optimize quaternions in isotropic mode."
            )

    def to_cuda(self):
        self.pre_act_quats = self.pre_act_quats.to(self.device)
        self.means = self.means.to(self.device)
        self.pre_act_scales = self.pre_act_scales.to(self.device)
        self.colours = self.colours.to(self.device)
        self.pre_act_opacities = self.pre_act_opacities.to(self.device)

        # [Q 1.3.1] NOTE: Uncomment spherical harmonics code for question 1.3.1
        self.spherical_harmonics = self.spherical_harmonics.to(self.device)

    def compute_cov_3D(self, quats: torch.Tensor, scales: torch.Tensor):
        """
        Computes the covariance matrices of 3D Gaussians using equation (6) of the 3DGS.

        Args:
            quats   :   A torch.Tensor of shape (N, 4) representing the rotation
                        components of 3D Gaussians in quaternion form.
            scales  :   If self.is_isotropic is True, scales is will be a torch.Tensor of shape (N, 1)
                        If self.is_isotropic is False, scales is will be a torch.Tensor of shape (N, 3).
                        Represents the scaling components of the 3D Gaussians.

        Returns:
            cov_3D  :   A torch.Tensor of shape (N, 3, 3)
        """
        # Quaternions should be already normalized in this function even Gaussians are anisotropic
        # because `apply_activations()` is called in `render()`.
        quats_norm = torch.linalg.vector_norm(quats, ord=2, dim=1)
        assert torch.allclose(
            quats_norm, torch.ones_like(quats_norm)
        ), "Not all rows have a Euclidean norm of 1"

        if self.is_isotropic:
            # In the case of isotropic, the quaternions are fixed value (not optimized by backpropagation).
            # Thus, the rotation matrices are always the same throughout the process.

            # Convert normalized quaternions to rotation matrices
            r_mats = quaternion_to_matrix(quats)  # (N, 3, 3)
            # Compute the scaling matrices from the scale vectors
            s_mats = torch.diag_embed(scales.repeat(1, 3))  # (N, 3, 3)
            cov_3D = (
                r_mats @ s_mats @ s_mats.transpose(1, 2) @ r_mats.transpose(1, 2)
            )  # (N, 3, 3)

        else:
            # Convert normalized quaternions to rotation matrices
            r_mats = quaternion_to_matrix(quats)  # (N, 3, 3)
            # Compute the scaling matrices from the scale vectors
            s_mats = torch.diag_embed(scales)  # (N, 3, 3)
            cov_3D = (
                r_mats @ s_mats @ s_mats.transpose(1, 2) @ r_mats.transpose(1, 2)
            )  # (N, 3, 3)

        return cov_3D

    def compute_cov_2D(
        self,
        means_3D: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        camera: PerspectiveCameras,
        img_size: Tuple,
    ):
        """
        Compute the covariance matrices of 2D Gaussians using equation (5) of the 3DGS.

        Args:
            quats       :   A torch.Tensor of shape (N, 4) representing the rotation
                            components of 3D Gaussians in quaternion form.
            scales      :   If self.is_isotropic is True, scales is will be a torch.Tensor of shape (N, 1)
                            If self.is_isotropic is False, scales is will be a torch.Tensor of shape (N, 3)
            camera      :   A pytorch3d PerspectiveCameras object
            img_size    :   A tuple representing the (width, height) of the image

        Returns:
            cov_3D  :   A torch.Tensor of shape (N, 3, 3)
        """
        J = self._compute_jacobian(
            means_3D=means_3D, camera=camera, img_size=img_size
        )  # (N, 2, 3)

        N = J.shape[0]
        # transform_matrix stands for transformation matrix that follows the same convention as Hartley & Zisserman.
        # I.e., for camera extrinsic parameters R (rotation) and T (translation), we map a 3D point X_world in world
        # coordinates to a point X_cam in camera coordinates with: X_cam = X_world R + T
        transform_matrix = (
            camera.get_world_to_view_transform().get_matrix()
        )  # (1, 4, 4)
        W = transform_matrix[..., :3, :3].repeat(N, 1, 1)  # (N, 3, 3)

        cov_3D = self.compute_cov_3D(quats=quats, scales=scales)  # (N, 3, 3)

        cov_2D = J @ W @ cov_3D @ W.transpose(1, 2) @ J.transpose(1, 2)  # (N, 2, 2)

        # Post processing to make sure that each 2D Gaussian covers atleast approximately 1 pixel
        cov_2D[:, 0, 0] += 0.3
        cov_2D[:, 1, 1] += 0.3

        return cov_2D

    @staticmethod
    def compute_means_2D(means_3D: torch.Tensor, camera: PerspectiveCameras):
        """
        Computes the means of the projected 2D Gaussians given the means of the 3D Gaussians.

        Args:
            means_3D    :   A torch.Tensor of shape (N, 3) representing the means of
                            3D Gaussians.
            camera      :   A pytorch3d PerspectiveCameras object.

        Returns:
            means_2D    :   A torch.Tensor of shape (N, 2) representing the means of
                            2D Gaussians.
        """
        # Note that means_2D have units of pixels. Hence, we must apply a
        # transformation that moves points in the world space to screen space.
        # `transform_points_screen` transforms points from PyTorch3D world/camera
        # space to screen space.
        means_2D = camera.transform_points_screen(means_3D)  # (N, 3)
        return means_2D[..., :-1]  # (N, 2)

    @staticmethod
    def invert_cov_2D(cov_2D: torch.Tensor):
        """
        Invert a 2D matrix to obtain its inverse.

        Args:
            cov_2D          :   A torch.Tensor of shape (N, 2, 2)

        Returns:
            cov_2D_inverse  :   A torch.Tensor of shape (N, 2, 2)
        """
        determinants = (
            cov_2D[:, 0, 0] * cov_2D[:, 1, 1] - cov_2D[:, 1, 0] * cov_2D[:, 0, 1]
        )
        determinants = determinants[:, None, None]  # (N, 1, 1)

        cov_2D_inverse = torch.zeros_like(cov_2D)  # (N, 2, 2)
        cov_2D_inverse[:, 0, 0] = cov_2D[:, 1, 1]
        cov_2D_inverse[:, 1, 1] = cov_2D[:, 0, 0]
        cov_2D_inverse[:, 0, 1] = -1.0 * cov_2D[:, 0, 1]
        cov_2D_inverse[:, 1, 0] = -1.0 * cov_2D[:, 1, 0]

        cov_2D_inverse = (1.0 / determinants) * cov_2D_inverse

        return cov_2D_inverse

    @staticmethod
    def evaluate_gaussian_2D(
        points_2D: torch.Tensor, means_2D: torch.Tensor, cov_2D_inverse: torch.Tensor
    ):
        """
        Computes the exponent (power) of 2D Gaussians.

        Args:
            points_2D       :   A torch.Tensor of shape (1, H*W, 2) containing the x, y points
                                corresponding to every pixel in an image. See function
                                compute_alphas in the class Scene to get more information
                                about how points_2D is created.
            means_2D        :   A torch.Tensor of shape (N, 1, 2) representing the means of
                                N 2D Gaussians.
            cov_2D_inverse  :   A torch.Tensor of shape (N, 2, 2) representing the
                                inverse of the covariance matrices of N 2D Gaussians.

        Returns:
            power           :   A torch.Tensor of shape (N, H*W) representing the computed
                                power of the N 2D Gaussians at every pixel location in an image.
        """
        # HINT: Refer to README for a relevant equation
        diff = (points_2D - means_2D).unsqueeze(-1)  # (N, H*W, 2, 1)
        cov_2D_inverse = cov_2D_inverse.unsqueeze(1)  # (N, 1, 2, 2)
        power = -0.5 * diff.transpose(-2, -1) @ cov_2D_inverse @ diff  # (N, H*W, 1, 1)
        return power.squeeze(-1).squeeze(-1)

    @staticmethod
    def apply_activations(pre_act_quats, pre_act_scales, pre_act_opacities):
        # Convert logscales to scales
        scales = torch.exp(pre_act_scales)

        # Normalize quaternions
        quats = torch.nn.functional.normalize(pre_act_quats)

        # Bound opacities between (0, 1)
        opacities = torch.sigmoid(pre_act_opacities)

        return quats, scales, opacities


class Scene:
    def __init__(self, gaussians: Gaussians):
        self.gaussians = gaussians
        self.device = self.gaussians.device

    def __repr__(self):
        return f"<Scene with {len(self.gaussians)} Gaussians>"

    def compute_depth_values(self, camera: PerspectiveCameras):
        """
        Computes the depth value of each 3D Gaussian.

        Args:
            camera  :   A pytorch3d PerspectiveCameras object.

        Returns:
            z_vals  :   A torch.Tensor of shape (N,) with the depth of each 3D Gaussian.
        """
        # We can use use the means of 3D Gaussians (N, 3), `self.gaussians.means`, and the camera
        # to calculate the depth wrt the camera space.
        # We transform (NOT project) input points from world to camera space. If camera is defined
        # in NDC space, the projected points are in NDC space. If camera is defined in screen space,
        # the projected points are in screen space.
        xyz_cam = camera.get_world_to_view_transform().transform_points(
            self.gaussians.means
        )  # (N, 3)
        z_vals = xyz_cam[..., -1]  # (N,)
        return z_vals

    def get_idxs_to_filter_and_sort(self, z_vals: torch.Tensor):
        """
        Given depth values of Gaussians, return the indices to depth-wise sort
        Gaussians and at the same time remove invalid Gaussians.

        You can see the function render to see how the returned indices will be used.
        You are required to create a torch.Tensor idxs such that by using them in the
        function render we can arrange Gaussians (or equivalently their attributes such as
        the mean) in ascending order of depth. You should also make sure to not include indices
        that correspond to Gaussians with depth value less than 0.

        idxs should be torch.Tensor of dtype int64 with length N (N <= M, where M is the
        total number of Gaussians before filtering)

        Please refer to the README file for more details.
        """
        sorted_idxs = torch.argsort(z_vals, descending=False)
        idxs = sorted_idxs[z_vals > 0]
        return idxs  # (N,)

    def compute_alphas(self, opacities, means_2D, cov_2D, img_size):
        """
        Given some parameters of N ordered Gaussians, this function computes
        the alpha values.

        Args:
            opacities   :   A torch.Tensor of shape (N,) with the opacity value
                            of each Gaussian.
            means_2D    :   A torch.Tensor of shape (N, 2) with the means
                            of the 2D Gaussians.
            cov_2D      :   A torch.Tensor of shape (N, 2, 2) with the covariances
                            of the 2D Gaussians.
            img_size    :   The (width, height) of the image to be rendered.


        Returns:
            alphas      :   A torch.Tensor of shape (N, H, W) with the computed alpha
                            values for each of the N ordered Gaussians at every
                            pixel location.
        """
        W, H = img_size

        # point_2D contains all possible pixel locations in an image
        xs, ys = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        points_2D = torch.stack((xs.flatten(), ys.flatten()), dim=1)  # (H*W, 2)
        points_2D = points_2D.to(self.device)

        points_2D = points_2D.unsqueeze(0)  # (1, H*W, 2)
        means_2D = means_2D.unsqueeze(1)  # (N, 1, 2)

        cov_2D_inverse = self.gaussians.invert_cov_2D(cov_2D=cov_2D)  # (N, 2, 2)
        power = self.gaussians.evaluate_gaussian_2D(
            points_2D=points_2D, means_2D=means_2D, cov_2D_inverse=cov_2D_inverse
        )  # (N, H*W)

        # Computing exp(power) with some post processing for numerical stability
        exp_power = torch.where(power > 0.0, 0.0, torch.exp(power))

        alphas = opacities.unsqueeze(1) * exp_power  # (N, H*W)
        alphas = torch.reshape(alphas, (-1, H, W))  # (N, H, W)

        # Post processing for numerical stability
        alphas = torch.minimum(alphas, torch.full_like(alphas, 0.99))
        alphas = torch.where(alphas < 1 / 255.0, 0.0, alphas)

        return alphas

    def compute_transmittance(
        self, alphas: torch.Tensor, start_transmittance: Optional[torch.Tensor] = None
    ):
        """
        Given the alpha values of N ordered Gaussians, this function computes
        the transmittance.

        The variable start_transmittance contains information about the transmittance
        at each pixel location BEFORE encountering the first Gaussian in the input.
        This variable is useful when computing transmittance in mini-batches because
        we would require information about the transmittance accumulated until the
        previous mini-batch to begin computing the transmittance for the current mini-batch.

        In case there were no previous mini-batches (or we are splatting in one-shot
        without using mini-batches), then start_transmittance will be None (since no Gaussians
        have been encountered so far). In this case, the code will use a starting
        transmittance value of 1.

        Args:
            alphas                  :   A torch.Tensor of shape (N, H, W) with the computed alpha
                                        values for each of the N ordered Gaussians at every
                                        pixel location.
            start_transmittance     :   Can be None or a torch.Tensor of shape (1, H, W). Please
                                        see the docstring for more information.

        Returns:
            transmittance           :   A torch.Tensor of shape (N, H, W) with the computed transmittance
                                        values for each of the N ordered Gaussians at every
                                        pixel location.
        """
        _, H, W = alphas.shape

        if start_transmittance is None:
            S = torch.ones((1, H, W), device=alphas.device, dtype=alphas.dtype)
        else:
            S = start_transmittance

        one_minus_alphas = 1.0 - alphas
        one_minus_alphas = torch.concat((S, one_minus_alphas), dim=0)  # (N+1, H, W)

        ### YOUR CODE HERE ###
        # HINT: Refer to README for a relevant equation.
        transmittance = torch.cumprod(one_minus_alphas, dim=0)[:-1, ...]  # (N, H, W)

        # Post processing for numerical stability
        transmittance = torch.where(
            transmittance < 1e-4, 0.0, transmittance
        )  # (N, H, W)

        return transmittance

    def splat(
        self,
        camera: PerspectiveCameras,
        means_3D: torch.tensor,
        z_vals: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        colours: torch.Tensor,
        opacities: torch.Tensor,
        img_size: Tuple = (256, 256),
        start_transmittance: Optional[torch.Tensor] = None,
    ):
        """
        Given N ordered (depth-sorted) 3D Gaussians (or equivalently in our case,
        the parameters of the 3D Gaussians like means, quats etc.), this function splats
        them to the image plane to render an RGB image, depth map and a silhouette map.

        Args:
            camera                  :   A pytorch3d PerspectiveCameras object.
            means_3D                :   A torch.Tensor of shape (N, 3) with the means
                                        of the 3D Gaussians.
            z_vals                  :   A torch.Tensor of shape (N,) with the depths
                                        of the 3D Gaussians. # TODO: Verify shape
            quats                   :   A torch.Tensor of shape (N, 4) representing the rotation
                                        components of 3D Gaussians in quaternion form.
            scales                  :   A torch.Tensor of shape (N, 1) (if isotropic) or
                                        (N, 3) (if anisotropic) representing the scaling
                                        components of 3D Gaussians.
            colours                 :   A torch.Tensor of shape (N, 3) with the colour contribution
                                        of each Gaussian.
            opacities               :   A torch.Tensor of shape (N,) with the opacity of each Gaussian.
            img_size                :   The (width, height) of the image.
            start_transmittance     :   Please see the docstring of the function compute_transmittance
                                        for information about this argument.

        Returns:
            image                   :   A torch.Tensor of shape (H, W, 3) with the rendered RGB colour image.
            depth                   :   A torch.Tensor of shape (H, W, 1) with the rendered depth map.
            mask                    :   A torch.Tensor of shape (H, W, 1) with the rendered silhouette map.
            final_transmittance     :   A torch.Tensor of shape (1, H, W) representing the transmittance at
                                        each pixel computed using the N ordered Gaussians. This will be useful
                                        for mini-batch splatting in the next iteration.
        """
        # Step 1: Compute 2D gaussian parameters
        means_2D = self.gaussians.compute_means_2D(
            means_3D=means_3D, camera=camera
        )  # (N, 2)
        cov_2D = self.gaussians.compute_cov_2D(
            means_3D=means_3D,
            quats=quats,
            scales=scales,
            camera=camera,
            img_size=img_size,
        )  # (N, 2, 2)

        # Step 2: Compute alpha maps for each gaussian
        alphas = self.compute_alphas(
            opacities=opacities, means_2D=means_2D, cov_2D=cov_2D, img_size=img_size
        )  # (N, H, W)

        # Step 3: Compute transmittance maps for each gaussian
        transmittance = self.compute_transmittance(
            alphas=alphas, start_transmittance=start_transmittance
        )  # (N, H, W)

        # Some unsqueezing to set up broadcasting for vectorized implementation.
        # You can selectively comment these out if you want to compute things
        # in a different way.
        z_vals = z_vals[:, None, None, None]  # (N, 1, 1, 1)
        alphas = alphas[..., None]  # (N, H, W, 1)
        colours = colours[:, None, None, :]  # (N, 1, 1, 3)
        transmittance = transmittance[..., None]  # (N, H, W, 1)

        # Step 4: Create image, depth and mask by computing the colours for each pixel.
        # Implement Eq.3 in the paper.
        image = torch.sum(colours * alphas * transmittance, dim=0)  # (H, W, 3)

        # Depth calculation inspired by the equation for colour.
        depth = torch.sum(z_vals * alphas * transmittance, dim=0)  # (H, W, 1)

        # Transmittance calculation inspired by the equation for colour.
        mask = torch.sum(alphas * transmittance, dim=0)  # (H, W, 1)

        final_transmittance = transmittance[-1, ..., 0].unsqueeze(0)  # (1, H, W)
        return image, depth, mask, final_transmittance

    def render(
        self,
        camera: PerspectiveCameras,
        per_splat: int = -1,
        img_size: Tuple = (256, 256),
        bg_colour: Tuple = (0.0, 0.0, 0.0),
    ):
        """
        Given a scene represented by N 3D Gaussians, this function renders the RGB
        colour image, the depth map and the silhouette map that can be observed
        from a given pytorch 3D camera.

        Args:
            camera      :   A pytorch3d PerspectiveCameras object.
            per_splat   :   Number of gaussians to splat in one function call. If set to -1,
                            then all gaussians in the scene are splat in a single function call.
                            If set to any other positive interger, then it determines the number of
                            gaussians to splat per function call (the last function call might splat
                            lesser number of gaussians). In general, the algorithm can run faster
                            if more gaussians are splat per function call, but at the cost of higher GPU
                            memory consumption.
            img_size    :   The (width, height) of the image to be rendered.
            bg_color    :   A tuple indicating the RGB colour that the background should have.

        Returns:
            image       :   A torch.Tensor of shape (H, W, 3) with the rendered RGB colour image.
            depth       :   A torch.Tensor of shape (H, W, 1) with the rendered depth map.
            mask        :   A torch.Tensor of shape (H, W, 1) with the rendered silhouette map.
        """
        bg_colour_ = torch.tensor(bg_colour)[None, None, :]  # (1, 1, 3)
        bg_colour_ = bg_colour_.to(self.device)

        # Globally sort gaussians according to their depth value
        z_vals = self.compute_depth_values(camera)
        idxs = self.get_idxs_to_filter_and_sort(z_vals)

        pre_act_quats = self.gaussians.pre_act_quats[idxs]
        pre_act_scales = self.gaussians.pre_act_scales[idxs]
        pre_act_opacities = self.gaussians.pre_act_opacities[idxs]
        z_vals = z_vals[idxs]
        means_3D = self.gaussians.means[idxs]

        # For questions 1.1, 1.2 and 1.3.2, use the below line of code for colours.
        colours = self.gaussians.colours[idxs]

        # [Q 1.3.1] For question 1.3.1, uncomment the below three lines to calculate the
        # colours instead of using self.gaussians.colours[idxs]. You may also comment
        # out the above line of code since it will be overwritten anyway.

        spherical_harmonics = self.gaussians.spherical_harmonics[idxs]
        gaussian_dirs = self.calculate_gaussian_directions(means_3D, camera)
        colours = colours_from_spherical_harmonics(spherical_harmonics, gaussian_dirs)

        # Apply activations
        quats, scales, opacities = self.gaussians.apply_activations(
            pre_act_quats, pre_act_scales, pre_act_opacities
        )

        if per_splat == -1:
            num_mini_batches = 1
        elif per_splat > 0:
            num_mini_batches = math.ceil(len(means_3D) / per_splat)
        else:
            raise ValueError("Invalid setting of per_splat")

        if num_mini_batches == 1:
            # Directly splat all gaussians onto the image

            image, depth, mask, _ = self.splat(
                camera, means_3D, z_vals, quats, scales, colours, opacities, img_size
            )

        else:
            # splat per_splat number of gaussians per iteration. This makes the
            # implementation more memory efficient but at the same time makes it slower.

            W, H = img_size
            D = means_3D.device
            start_transmittance = torch.ones((1, H, W), dtype=torch.float32, device=D)
            image = torch.zeros((H, W, 3), dtype=torch.float32, device=D)
            depth = torch.zeros((H, W, 1), dtype=torch.float32, device=D)
            mask = torch.zeros((H, W, 1), dtype=torch.float32, device=D)

            for b_idx in range(num_mini_batches):
                quats_ = quats[b_idx * per_splat : (b_idx + 1) * per_splat]
                scales_ = scales[b_idx * per_splat : (b_idx + 1) * per_splat]
                z_vals_ = z_vals[b_idx * per_splat : (b_idx + 1) * per_splat]
                colours_ = colours[b_idx * per_splat : (b_idx + 1) * per_splat]
                means_3D_ = means_3D[b_idx * per_splat : (b_idx + 1) * per_splat]
                opacities_ = opacities[b_idx * per_splat : (b_idx + 1) * per_splat]

                image_, depth_, mask_, start_transmittance = self.splat(
                    camera=camera,
                    means_3D=means_3D_,
                    z_vals=z_vals_,
                    quats=quats_,
                    scales=scales_,
                    colours=colours_,
                    opacities=opacities_,
                    img_size=img_size,
                    start_transmittance=start_transmittance,
                )
                assert image.shape == image_.shape and image.device == image_.device
                assert depth.shape == depth_.shape
                assert mask.shape == mask_.shape
                # In-place addition to avoid creating new tensors
                image.add_(image_)
                depth.add_(depth_)
                mask.add_(mask_)

        image = mask * image + (1.0 - mask) * bg_colour_

        return image, depth, mask

    def calculate_gaussian_directions(self, means_3D, camera):
        """
        [Q 1.3.1] Calculates the world frame direction vectors that point from the
        camera's origin to each 3D Gaussian.

        Args:
            means_3D        :   A torch.Tensor of shape (N, 3) with the means
                                of the 3D Gaussians.
            camera          :   A pytorch3d PerspectiveCameras object.

        Returns:
            gaussian_dirs   :   A torch.Tensor of shape (N, 3) representing the direction vector
                                that points from the camera's origin to each 3D Gaussian.
        """
        # HINT: Think about how to get the camera origin in the world frame.
        # HINT: Do not forget to normalize the computed directions.
        camera_center = camera.get_camera_center()  # (1, 3)
        camera_centers = camera_center.repeat(means_3D.shape[0], 1)  # (N, 3)
        gaussian_dirs = means_3D - camera_centers  # (N, 3)

        gaussian_dirs = torch.nn.functional.normalize(gaussian_dirs, p=2, dim=1)
        return gaussian_dirs