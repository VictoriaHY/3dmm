import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from pytorch3d.io import load_obj
import imageio
import numpy as np


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def get_points_renderer(
    image_size=512, device=None, radius=0.001, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
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
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

def unproj_depth(u, v, depth, K):
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]

    # print('cx',cx)
    # print('cy',cy)

    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth

    # d = torch.ones_like(x)

    xyz = torch.stack([y, x, depth], dim=-1)
    # print(xyz.shape)
    
    return xyz


def unproject_depth_image(image, depth, K, mask, device=None):
    """
    Unprojects a depth image into a 3D point cloud.

    Args:
        image (torch.Tensor): A square image to unproject (S, S, 3).
        mask (torch.Tensor): A binary mask for the image (S, S).
        depth (torch.Tensor): The depth map of the image (S, S).
        camera: The Pytorch3D camera to render the image.
    
    Returns:
        points (torch.Tensor): The 3D points of the unprojected image (N, 3).
        rgba (torch.Tensor): The rgba color values corresponding to the unprojected
            points (N, 4).
    """
    # device = camera.device
    if device is None:
        device = get_device()
    # assert image.shape[0] == image.shape[1], "Image must be square."
    # image_shape = image.shape[0]
    ndc_pixel_coordinates = torch.linspace(0, 640-1, image.shape[0])
    ndc_pixel_coordinates1 = torch.linspace(0, 360-1, image.shape[1])
    Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates1)
    # depth = torch.ones_like(torch.tensor(depth, dtype=torch.float))
    # print(Y, X)
    depth = torch.tensor(depth, dtype=torch.float)
    xy_depth = torch.dstack([Y, X, depth])
    # print(xy_depth.shape)
    # print(xy_depth)
    # points1 = camera.unproject_points(
    #     xy_depth.to(device), in_ndc=False, from_ndc=False, world_coordinates=True,
    # )
    # print('points1', points1)
    points = unproj_depth(Y, X, depth, K)
    # print('points2', points)
    # print('delta', (points-points1).sum())
    
    # mask = torch.ones_like(points)[...,0]
    
    # imageio.imsave('depth2.jpg',np.array(points[:,2]).reshape(640,360))
    points = points[mask > 0.5]
    # points = points.reshape(-1,3)
    
    rgb = torch.tensor(image, dtype=torch.float)[mask > 0.5].to(device)
    # rgb = torch.tensor(image, dtype=torch.float).reshape(-1,3).to(device)
    

    # For some reason, the Pytorch3D compositor does not apply a background color
    # unless the pointcloud is RGBA.
    alpha = torch.ones_like(rgb)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=1)

    return points, rgb


def unproject_depth_image1(image, depth, K, mask, device=None):
    """
    Unprojects a depth image into a 3D point cloud.

    Args:
        image (torch.Tensor): A square image to unproject (S, S, 3).
        mask (torch.Tensor): A binary mask for the image (S, S).
        depth (torch.Tensor): The depth map of the image (S, S).
        camera: The Pytorch3D camera to render the image.
    
    Returns:
        points (torch.Tensor): The 3D points of the unprojected image (N, 3).
        rgba (torch.Tensor): The rgba color values corresponding to the unprojected
            points (N, 4).
    """
    # device = camera.device
    if device is None:
        device = get_device()
    # assert image.shape[0] == image.shape[1], "Image must be square."
    # image_shape = image.shape[0]
    ndc_pixel_coordinates = torch.linspace(0, 640-1, image.shape[0])
    ndc_pixel_coordinates1 = torch.linspace(0, 360-1, image.shape[1])
    Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates1)
    # depth = torch.ones_like(torch.tensor(depth, dtype=torch.float))
    # print(Y, X)
    depth = torch.tensor(depth, dtype=torch.float)
    # xy_depth = torch.dstack([Y, X, depth])
    # print(xy_depth.shape)
    # print(xy_depth)
    # points1 = camera.unproject_points(
    #     xy_depth.to(device), in_ndc=False, from_ndc=False, world_coordinates=True,
    # )
    # print('points1', points1)
    points = unproj_depth(Y, X, depth, K)
    # print('points2', points)
    # print('delta', (points-points1).sum())
    
    # mask = torch.ones_like(points)[...,0]
    
    # imageio.imsave('depth2.jpg',np.array(points[:,2]).reshape(640,360))
    points = points[mask[:,1], mask[:,0], :]
    # points = points.reshape(-1,3)
    
    rgb = torch.tensor(image, dtype=torch.float)[mask[:,1], mask[:,0], :].to(device)
    # rgb = torch.tensor(image, dtype=torch.float).reshape(-1,3).to(device)
    

    # For some reason, the Pytorch3D compositor does not apply a background color
    # unless the pointcloud is RGBA.
    alpha = torch.ones_like(rgb)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=1)

    return points, rgb


