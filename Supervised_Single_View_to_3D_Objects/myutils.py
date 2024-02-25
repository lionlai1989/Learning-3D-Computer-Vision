import torch
import numpy as np
import torch.nn.functional as F
import pytorch3d

from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.mesh.renderer import MeshRenderer
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.mesh.textures import TexturesVertex, Textures
from pytorch3d.renderer.mesh.shader import SoftPhongShader, HardPhongShader
from pytorch3d.renderer.points.renderer import PointsRenderer
from pytorch3d.renderer.points.rasterizer import (
    PointsRasterizer,
    PointsRasterizationSettings,
)
from pytorch3d.renderer.points.compositor import AlphaCompositor
from pytorch3d.renderer.implicit.renderer import VolumeRenderer
from pytorch3d.renderer.implicit.raysampling import NDCMultinomialRaysampler
from pytorch3d.renderer.implicit.raymarching import EmissionAbsorptionRaymarcher
from pytorch3d.renderer.cameras import (
    look_at_view_transform,
    look_at_rotation,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
)
from pytorch3d.structures import Meshes, Pointclouds, Volumes

XMIN = -0.5  # right (neg is left)
XMAX = 0.5  # right
YMIN = -0.5  # down (neg is up)
YMAX = 0.5  # down
ZMIN = -0.5  # forward
ZMAX = 0.5  # forward


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_pointcloud_renderer(
    image_size, device, radius=0.01, background_color=(0.5, 0.5, 0.5)
):
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def add_texture_to_mesh(mesh_gt):
    vertices = mesh_gt.verts_list()
    faces = mesh_gt.faces_list()

    # convert list to tensor
    vertices = torch.cat(vertices)
    faces = torch.cat(faces)

    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    color = [0.7, 0.7, 1]

    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color).to(get_device())  # (1, N_v, 3)

    mesh_gt = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )

    return mesh_gt

def render_rotating_meshes(meshes, image_size, device):
    mesh_input = add_texture_to_mesh(meshes)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.5, -4.0]], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    batch_size = 120
    images = []
    for azim in torch.linspace(0, 360, batch_size):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=2.7, elev=0, azim=azim)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh_input, cameras=cameras, lights=lights)
        img = rend.detach().cpu().numpy().clip(0, 1)[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        img *= 255
        images.append(img.astype(np.uint8))

    return images

def render_mesh(mesh_input, device, image_size=256):
    mesh_input = add_texture_to_mesh(mesh_input)

    lights = pytorch3d.renderer.PointLights(location=[[0, 0.5, -4.0]], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    R, T = pytorch3d.renderer.look_at_view_transform(dist=2.7, elev=30, azim=120)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh_input, cameras=cameras, lights=lights)

    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)

def render_rotating_pointclouds(points, image_size, device):
    renderer = get_pointcloud_renderer(image_size=image_size, device=device)
    rgb = torch.rand(points.shape).to(device)
    pointcloud = pytorch3d.structures.Pointclouds(points=points, features=rgb)

    batch_size = 120
    images = []
    for azim in torch.linspace(0, 360, batch_size):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=2.7, elev=0, azim=azim)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(pointcloud, cameras=cameras)
        img = rend.detach().cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        img *= 255
        images.append(img.astype(np.uint8))
    return images

def render_pointcloud(points, device, image_size=256):
    renderer = get_pointcloud_renderer(image_size=image_size, device=device)

    rgb = torch.rand(points.shape).to(device)

    pointcloud = pytorch3d.structures.Pointclouds(points=points, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=2.7, elev=30, azim=120)
    # R = R @ torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(pointcloud, cameras=cameras)
    return rend.detach().cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

def render_rotating_voxels(voxels, device, image_size=256):
    mesh = pytorch3d.ops.cubify(voxels, thresh=0.5)
    mesh = mesh.to(device)
    vertices = mesh.verts_packed().unsqueeze(0)
    faces = mesh.faces_packed().unsqueeze(0)
    texture = torch.ones_like(vertices) * 0.5  # (1, N_v, 3)
    textures = pytorch3d.renderer.TexturesVertex(texture)

    mesh = pytorch3d.structures.Meshes(vertices, faces, textures=textures)
    mesh = mesh.to(device)

    lights = pytorch3d.renderer.PointLights(location=[[0, 0.5, -4.0]], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)


    batch_size = 120
    images = []
    for azim in torch.linspace(0, 360, batch_size):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=2.7, elev=0, azim=azim)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        img = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
        img *= 255
        images.append(img.astype(np.uint8))

    return images


def render_voxels_as_mesh(voxels, device, image_size=256):

    # print(f"shape of voxels {voxels.shape}") # (1, 32, 32, 32)
    mesh = pytorch3d.ops.cubify(voxels, thresh=0.5)
    mesh = mesh.to(device)
    vertices = mesh.verts_packed().unsqueeze(0)
    faces = mesh.faces_packed().unsqueeze(0)
    texture = torch.ones_like(vertices) * 0.5  # (1, N_v, 3)
    textures = pytorch3d.renderer.TexturesVertex(texture)

    mesh = pytorch3d.structures.Meshes(vertices, faces, textures=textures)
    mesh = mesh.to(device)

    lights = pytorch3d.renderer.PointLights(location=[[0, 0.5, -4.0]], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    R, T = pytorch3d.renderer.look_at_view_transform(dist=2.7, elev=0, azim=120)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)

    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)


# def colorize_voxels(voxels):

#     # Ensure the color is a tensor
#     color_tensor = torch.tensor(color, dtype=torch.float32)

#     # Check if the color_tensor is of the correct shape and type
#     if color_tensor.shape != (3,) or color_tensor.dtype != torch.float32:
#         raise ValueError("Color must be a tensor of shape (3,) and dtype torch.float32")

#     # Initialize an output tensor of zeros with the desired shape [3, D, H, W]
#     colored_voxels = torch.zeros((3, *voxels.shape[1:]), dtype=torch.float32)

#     # Apply the color to each voxel that is True (non-zero)
#     for i in range(3):  # Iterate over the RGB color channels
#         colored_voxels[i][voxel_tensor[0] == 1] = color_tensor[i]

#     return colored_voxels

# def render_voxel(voxels, device, image_size):
#     """Returns primitive images without any postprocessing."""

#     # voxels: batch x height x width x depth
#     print("voxels: ", voxels.shape)

#     colors = torch.rand(voxels.shape).to(device=device)
#     raysampler = NDCMultinomialRaysampler(
#         image_width=image_size,
#         image_height=image_size,
#         n_pts_per_ray=500,  # maximum value if bigger, memory error
#         min_depth=0.01,
#         max_depth=10,
#     )
#     renderer = VolumeRenderer(
#         raysampler=raysampler,
#         raymarcher=EmissionAbsorptionRaymarcher(),
#     )
#     R, T = look_at_view_transform(dist=2.7, elev=0, azim=120)
#     camera = FoVPerspectiveCameras(
#         R=R,
#         T=T,
#         znear=0.01,
#         zfar=100,
#         aspect_ratio=1,
#         fov=45,
#         device=device,
#     )
#     volumes = Volumes(
#         densities=[voxels],
#         features=[colors],
#         voxel_size=0.05,
#     )
#     rendered_images, rendered_silhouettes = renderer(cameras=camera, volumes=volumes)
#     # Is there a difference between
#     # .cpu().detach().numpy()
#     # .detach().cpu().numpy()
#     img = rendered_images[0, ..., :3].cpu().detach().numpy()  # HxWx1

#     return img # np.concatenate([img, img, img], axis=2)


def voxelize_xyz(xyz_ref, Z, Y, X, already_mem=False):
    B, N, D = list(xyz_ref.shape)
    assert D == 3
    if already_mem:
        xyz_mem = xyz_ref
    else:
        xyz_mem = Ref2Mem(xyz_ref, Z, Y, X)
    vox = get_occupancy(xyz_mem, Z, Y, X)
    return vox


def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:, :, 0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    xyz2 = xyz2[:, :, :3]
    return xyz2


def eye_4x4(B, device="cpu"):
    rt = torch.eye(4, device=torch.device(device)).view(1, 4, 4).repeat([B, 1, 1])
    return rt


def Ref2Mem(xyz, Z, Y, X):
    # xyz is B x N x 3, in ref coordinates
    # transforms velo coordinates into mem coordinates
    B, N, C = list(xyz.shape)
    mem_T_ref = get_mem_T_ref(B, Z, Y, X)
    xyz = apply_4x4(mem_T_ref, xyz)
    return xyz


def get_occupancy(xyz, Z, Y, X):
    # xyz is B x N x 3 and in mem coords
    # we want to fill a voxel tensor with 1's at these inds
    B, N, C = list(xyz.shape)
    assert C == 3

    # these papers say simple 1/0 occupancy is ok:
    #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf
    #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
    # cont fusion says they do 8-neighbor interp
    # voxelnet does occupancy but with a bit of randomness in terms of the reflectance value i think

    inbounds = get_inbounds(xyz, Z, Y, X, already_mem=True)
    x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
    mask = torch.zeros_like(x)
    mask[inbounds] = 1.0

    # set the invalid guys to zero
    # we then need to zero out 0,0,0
    # (this method seems a bit clumsy)
    x = x * mask
    y = y * mask
    z = z * mask

    x = torch.round(x)
    y = torch.round(y)
    z = torch.round(z)
    x = torch.clamp(x, 0, X - 1).int()
    y = torch.clamp(y, 0, Y - 1).int()
    z = torch.clamp(z, 0, Z - 1).int()

    x = x.view(B * N)
    y = y.view(B * N)
    z = z.view(B * N)

    dim3 = X
    dim2 = X * Y
    dim1 = X * Y * Z

    # base = torch.from_numpy(np.concatenate([np.array([i*dim1]) for i in range(B)]).astype(np.int32))
    # base = torch.range(0, B-1, dtype=torch.int32, device=torch.device('cpu'))*dim1
    base = torch.arange(0, B, dtype=torch.int32, device=torch.device("cpu")) * dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B * N)

    vox_inds = base + z * dim2 + y * dim3 + x
    voxels = torch.zeros(B * Z * Y * X, device=torch.device("cpu")).float()
    voxels[vox_inds.long()] = 1.0
    # zero out the singularity
    voxels[base.long()] = 0.0
    voxels = voxels.reshape(B, 1, Z, Y, X)
    # B x 1 x Z x Y x X
    return voxels


def matmul2(mat1, mat2):
    return torch.matmul(mat1, mat2)


def Mem2Ref(xyz_mem, Z, Y, X, device="cpu"):
    # xyz is B x N x 3, in mem coordinates
    # transforms mem coordinates into ref coordinates
    B, N, C = list(xyz_mem.shape)
    # st()
    ref_T_mem = get_ref_T_mem(B, Z, Y, X, device=device)
    xyz_ref = apply_4x4(ref_T_mem, xyz_mem)
    return xyz_ref


def get_ref_T_mem(B, Z, Y, X, device="cpu"):
    mem_T_ref = get_mem_T_ref(B, Z, Y, X, device=device)
    # note safe_inverse is inapplicable here,
    # since the transform is nonrigid
    ref_T_mem = mem_T_ref.inverse()
    return ref_T_mem


def get_mem_T_ref(B, Z, Y, X, device="cpu"):
    # sometimes we want the mat itself
    # note this is not a rigid transform

    # for interpretability, let's construct this in two steps...

    # translation
    center_T_ref = eye_4x4(B, device=device)
    center_T_ref[:, 0, 3] = -XMIN
    center_T_ref[:, 1, 3] = -YMIN
    center_T_ref[:, 2, 3] = -ZMIN

    VOX_SIZE_X = (XMAX - XMIN) / float(X)
    VOX_SIZE_Y = (YMAX - YMIN) / float(Y)
    VOX_SIZE_Z = (ZMAX - ZMIN) / float(Z)

    # scaling
    mem_T_center = eye_4x4(B, device=device)
    mem_T_center[:, 0, 0] = 1.0 / VOX_SIZE_X
    mem_T_center[:, 1, 1] = 1.0 / VOX_SIZE_Y
    mem_T_center[:, 2, 2] = 1.0 / VOX_SIZE_Z
    mem_T_ref = matmul2(mem_T_center, center_T_ref)

    return mem_T_ref


def get_inbounds(xyz, Z, Y, X, already_mem=False):
    # xyz is B x N x 3
    if not already_mem:
        xyz = Ref2Mem(xyz, Z, Y, X)

    x = xyz[:, :, 0]
    y = xyz[:, :, 1]
    z = xyz[:, :, 2]

    x_valid = (x > -0.5).byte() & (x < float(X - 0.5)).byte()
    y_valid = (y > -0.5).byte() & (y < float(Y - 0.5)).byte()
    z_valid = (z > -0.5).byte() & (z < float(Z - 0.5)).byte()

    inbounds = x_valid & y_valid & z_valid
    return inbounds.bool()
