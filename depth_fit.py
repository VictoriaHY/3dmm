

from importlib.util import decode_source
import subprocess, os, json
from weakref import ref
import pandas as pd
import h5py

import numpy as np
# import copy
import cv2
# import struct
import os

import imageio
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np

from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image

import pickle

import mcubes
import zlib
import struct
import torch.nn as nn

from pytorch3d.io import load_obj

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

# import cython
import numba as nb

model_bin_path = "./FaceMorphModel_199.bin"

# Depth data parameters
w = 640
h = 360
data_folder = '.'
depth_path = data_folder + '/depth_2.xyz'
video_path = data_folder + '/output_2.mov'
output_video_path = data_folder + '/rgbd_2_bfs.mp4'

import wandb
USE_WANDB = False
import torch.nn.functional as F

@nb.jit()
def corres(corres, points1, b1, b2, num):
    for i in range(b1, b2):
        for j in range(num):
            corres[i, points1[i-b1,j,1].long(), points1[i-b1,j,0].long(),0] = j/num*2-1
    corres[:,:,:,1] = 0
    return corres

class Model(nn.Module):
    def __init__(self, mean, pca, variance, renderer, key_ref, key_side_ref, key_keypoint, frame_num, ref_depth):
        super().__init__()
        self.mean = mean
        self.pca = pca
        self.variance = variance
        self.key_keypoint = key_keypoint
        self.ref_depth = ref_depth
        
        self.inner = torch.tensor([8, 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67])
        self.side = torch.tensor([0,1,2,3,4,5,6,7, 9,10,11,12,13,14,15,16])
        
        self.batchsize = frame_num
        
        self.device = mean.device
        self.renderer = renderer
        # self.face = face
        
        # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        # image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        self.key_ref = key_ref
        self.key_side_ref = key_side_ref
        # self.register_buffer('image_ref', image_ref)
        
        # Create an optimizable parameter for the x, y, z position of the camera. 
        # self.camera_position = nn.Parameter(
        #     torch.from_numpy(np.array([0.0,  0.0, 1.0], dtype=np.float32)).to(points.device))
        # self.camera_position = nn.Parameter(
        #     150*torch.ones((frame_num, 3), dtype=torch.float32, device=self.device))
        R = torch.zeros((frame_num, 3), dtype=torch.float32, device=self.device)
        R[:,2] = np.pi
        R[:,1] = np.pi
        self.R = nn.Parameter(R)
        T = torch.zeros((frame_num, 3), dtype=torch.float32, device=self.device)
        T[:,2] =300
        self.T = nn.Parameter(T)

        # self.param = nn.Parameter(
        #     torch.zeros(self.pca.shape[1], dtype=torch.float32, device=self.device))
        self.param = nn.Parameter(torch.zeros(self.pca.shape[1], dtype=torch.float32, device=self.device))
        # self.param = 0
        self.criterion = torch.nn.MSELoss()

        self.camera = pytorch3d.renderer.PerspectiveCameras(focal_length = ((436.16/180, 436.16/180), ), principal_point=((-(320.08-320)/320, -(179.22-180)/180.),), device=self.device)
        p_matrix = np.array([436.16, 0.0, 179.22, \
                            0.0, 436.16, 320.08, \
                            0.0, 0.0, 1.0], dtype=np.float32)

        p_matrix = np.tile(p_matrix.reshape(1, 3, 3), [self.batchsize, 1, 1])
        reverse_z = np.tile(np.reshape(np.array([1.0,0,0,0,1,0,0,0,-1.0], dtype=np.float32),[1,3,3]),
                            [self.batchsize,1,1])
        
        self.p_matrix = torch.tensor(p_matrix, device=self.device)
        self.reverse_z = torch.tensor(reverse_z, device=self.device)
        self.num = 53149
        # self.num = 11510
        self.num1 = 68
        
        self.corresp = torch.ones((frame_num, 640, 360, 2), device=self.device)
        self.m = (self.corresp>0)

    @staticmethod
    def Compute_rotation_matrix(angles):
        n_b = angles.size(0)
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        rotXYZ = torch.eye(3).view(1, 3, 3).repeat(n_b * 3, 1, 1).view(3, n_b, 3, 3)

        if angles.is_cuda: rotXYZ = rotXYZ.cuda()

        rotXYZ[0, :, 1, 1] = cosx
        rotXYZ[0, :, 1, 2] = -sinx
        rotXYZ[0, :, 2, 1] = sinx
        rotXYZ[0, :, 2, 2] = cosx
        rotXYZ[1, :, 0, 0] = cosy
        rotXYZ[1, :, 0, 2] = siny
        rotXYZ[1, :, 2, 0] = -siny
        rotXYZ[1, :, 2, 2] = cosy
        rotXYZ[2, :, 0, 0] = cosz
        rotXYZ[2, :, 0, 1] = -sinz
        rotXYZ[2, :, 1, 0] = sinz
        rotXYZ[2, :, 1, 1] = cosz

        rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])

        return rotation #.permute(0, 2, 1)

    def project(self, points, batchsize):
        
        # batchsize = self.batchsize
        camera_pos = torch.tensor([0.0,0.0,0.0], device=self.device).reshape(1, 1, 3)

        # tensor.reshape(constant([0.0,0.0,10.0]),[1,1,3])
        
        points = torch.matmul(points, self.reverse_z[:batchsize,:,:]) + camera_pos
        aug_projection = torch.matmul(points,self.p_matrix[:batchsize,:,:].permute((0,2,1)))

        face_projection = aug_projection[:,:,:]/torch.reshape(aug_projection[:,:,2],[batchsize,-1,1])
        return face_projection#, torch.reshape(aug_projection[:,:,2],[batchsize,-1,1])
    
    # def get_depth(self, points):



    def forward(self, d=0, b1=0, b2=0, update=False):
        
        # Render the image using the updated camera position. Based on the new position of the 
        # camera we calculate the rotation and translation matrices
        # R = pytorch3d.renderer.look_at_rotation(self.camera_position, device=self.device)  # (1, 3, 3)
        # T = -torch.bmm(R.transpose(1, 2), self.camera_position[:, :, None])[:, :, 0]   # (1, 3)
        # print(self.Compute_rotation_matrix(torch.tensor([[0,0,np.pi/3]])))
        R1 = self.Compute_rotation_matrix(self.R)
        T1 = self.T

        points = self.mean + (self.pca @ (self.param * torch.sqrt(self.variance)).reshape(-1,1)).reshape(-1,3)
        # points = self.mean
        
        points = points.unsqueeze(0)
        # index = torch.randint(0, points.shape[1]-1, (568,), device=self.device)
        # index[:68] = self.key_keypoint
        # points = points[:,index,:]
        points_k = points[:, self.key_keypoint, :]
        points_k = torch.bmm(torch.broadcast_to(points_k, (R1.shape[0],self.num1, 3)), R1) + torch.broadcast_to(T1.unsqueeze(1), (T1.shape[0],self.num1, 3))
        # points_k = points_k*10
        points_k = self.project(points_k, self.batchsize)
        points_k = points_k[:,:,:2]
        # points_k = points[:, self.key_keypoint,:2]
        # hy: depth[i,j,k] = depth[i, points[i,0], points[i,1]]


        
        loss_k = torch.mean((points_k[:, self.inner, :] - self.key_ref) ** 2)
        loss_k1 = torch.mean((points_k[:, self.side, :] - self.key_side_ref) ** 2)

        # print(self.param.shape)
        loss_reg = torch.square(self.param).mean()
        loss_d = None
        loss_d1=None
        mask=None
        pred_d = None
        # print(loss_d.shape)
        # print(loss_d>1000)
        if b2>0:
            # print(loss_k.data)
            # r_depth = torch.zeros((self.batchsize, 500), dtype=torch.float32, device=self.device)
            # for i in range(self.batchsize):
            #     r_depth[i, :] = self.ref_depth[i, points[i,68:,1].long(), points[i,68:,0].long(),0]
            
            points = torch.bmm(torch.broadcast_to(points, (b2-b1,self.num, 3)), R1[b1:b2,:,:]) + torch.broadcast_to(T1[b1:b2,:].unsqueeze(1), (b2-b1,self.num, 3))
            # points = points*10
            points1 = self.project(points, b2-b1)
            
            # if not torch.all(points1[:,:,0] == torch.clip(points1[:,:,0],0,360-1)):
            #     print(points1[:,:,0].min())
            #     print(points1[:,:,0].max())
            #     print(torch.sum(points1[:,:,0] != torch.clip(points1[:,:,0],0,360-1)))
            #     # assert(torch.all(points1[:,:,0] == torch.clip(points1[:,:,0],0,360-1)))
            # points1[:,:,0] = torch.clip(points1[:,:,0],0,360-1)
            # if not torch.all(points1[:,:,1] == torch.clip(points1[:,:,1],0,640-1)):
            #     print(points1[:,:,1].min())
            #     print(points1[:,:,1].max())
            #     print(torch.sum(points1[:,:,1] != torch.clip(points1[:,:,1],0,640-1)))
            #     # assert(torch.all(points1[:,:,1] == torch.clip(points1[:,:,1],0,640-1)))
            # points1[:,:,1] = torch.clip(points1[:,:,1],0,640-1)

            m0 = ((points1[:,:,1]>=0) & (points1[:,:,1]<=640-1)) & ((points1[:,:,0]>=0) & (points1[:,:,0]<=360-1))
            m0 = m0[:,:,None].detach()
            # print(points.shape)
            # print(m0.shape)
            points1 = points1 * m0
            
            
            if update:
                # %cython
                # cdef int i,j
                # for i in xrange(b1, b2):
                #     for j in xrange(self.num):
                #         self.corresp[i, points1[i-b1,j,1].long(), points1[i-b1,j,0].long(),0] = j/self.num*2-1
                        
                # for i in range(b1, b2):
                #     for j in range(self.num):
                #         self.corresp[i, points1[i-b1,j,1].long(), points1[i-b1,j,0].long(),0] = j/self.num*2-1
                # self.corresp[:,:,:,1] = 0
                self.corresp = corres(self.corresp, points1, b1, b2, self.num)
            self.m = (self.corresp>=-1) & (self.corresp<1)
            pred_d = torch.zeros((b2-b1, 640, 360, 1), dtype=torch.float32, device=self.device)
            # print('w',points1[:,:,1].max(), points1[:,:,1].min())
            # print('h',points1[:,:,0].max(), points1[:,:,0].min())
            # for i in range(0,b2-b1):
            #     pred_d[i,:,:,0] = points[i, self.corresp[b1+i, :,:,0].long(), 2]
            #     # pred_d[i,points1[i,:,1].long(), points1[i,:,0].long(),0] = points[i,:,2]#.double()
            
            pred_d = F.grid_sample(points1[:,None,:,1:2], self.corresp[b1:b2, :,:,:], mode='nearest')
            # print(pred_d.shape)
            
            # print(pred_d[0:10,0, 315:325, 175:185])
            # print(ref_depth[0:10, 315:325, 175:185, 0])
            pred_d = pred_d*self.m[b1:b2,None, : , :, 0]
            pred_d = pred_d.squeeze(1)
            # print(self.m.shape)
            # print(pred_d.shape)

            # print(points[:,68:,2].shape)
            # print(r_depth.shape)
            # loss_d = (points[:,68:,2] - r_depth)**2
            # print(points1[:,None,:,1:2].max())
            # print(pred_d.max())
            # print(pred_d.min())
            # if d>0:
            loss_d = ((pred_d - self.ref_depth[b1:b2,:,:,0]*self.m[b1:b2, : , :, 0])**2)
            # print(loss_d.max())
            # print('pred',pred_d[0,315:325, 175:185])
            # print('ref', ref_depth[0, 315:325, 175:185, 0])
            # print(loss_d[0, 315:325, 175:185])
            mask = (loss_d<d) & (loss_d>1e-6) & (pred_d>0)
            
            
            # d = torch.log(pred_d[mask]) - torch.log(self.ref_depth[b1:b2,:,:,:][mask])
            
            # loss_d1 = torch.sqrt((d**2).mean() - 0.85*(d.mean()**2))
            # loss_d = torch.where(loss_d>1000, 0., loss_d)
            
            # loss_d = loss_d.masked_fill(loss_d>1000, 0)
            
            loss_d[(loss_d>d) | (pred_d<1e-5)] = 0
            # print(loss_d.shape)
            loss_d = (loss_d.sum(dim=(1,2))/(mask.sum(dim=(1,2))+1)).mean()
        
            # print(loss_d)
        # loss = loss1 + 0.1*loss2
        # loss = self.criterion(points[:,:,:,0], self.key_ref)
        return loss_k, loss_k1, loss_reg, loss_d, R1, T1, points_k, mask, pred_d

def load_BFM_2019(fname="./model2017-1_bfm_nomouth.h5"):

    with h5py.File(fname, 'r') as f:
        shape_mean = f['shape']['model']['mean'][:]
        shape_pcaBasis = f['shape']['model']['pcaBasis'][:]
        shape_pcaVariance = f['shape']['model']['pcaVariance'][:]

        expression_mean = f['expression']['model']['mean'][:]
        expression_pcaBasis = f['expression']['model']['pcaBasis'][:]
        expression_pcaVariance = f['expression']['model']['pcaVariance'][:]

        color_mean = f['color']['model']['mean'][:]
        color_pcaBasis = f['color']['model']['pcaBasis'][:]
        color_pcaVariance = f['color']['model']['pcaVariance'][:]

        faces = f['shape']['representer']['cells'][:].transpose(1,0)
        faces = torch.tensor(faces)

        print(shape_mean.shape)
        print(shape_pcaBasis.shape)
        print(shape_pcaVariance.shape)
        print(faces.shape)

        return {'shape_mean': shape_mean, 'shape_pcaBasis': shape_pcaBasis, 'shape_pcaVariance': shape_pcaVariance,
            'expression_mean': expression_mean, 'expression_pcaBasis': expression_pcaBasis, 'expression_pcaVariance': expression_pcaVariance,
            'color_mean': color_mean, 'color_pcaBasis': color_pcaBasis, 'color_pcaVariance': color_pcaVariance,
            'face': faces}
        # return {'verts': v_bfm, 'color': c_bfm, 'shape_coeffs': shape_coeffs, 'exp_coeffs': exp_coeffs, 'color_coeffs': color_coeffs}

def load_facewarehouse_bin(bin_path="./FaceMorphModel_199.bin"):
    with open(bin_path, "rb") as f:
        # Basic parameters
        vecNum = struct.unpack("i", f.read(4))[0]
        ptNum = struct.unpack("i", f.read(4))[0]

        # Average face
        mAvergPosList = []
        for i in range(ptNum):
            x,y,z = struct.unpack("fff", f.read(12))
            mAvergPosList.append(np.array([x,y,z]))
        mAvergPos = np.stack(mAvergPosList)
        
        # Eigen values
        mEigenValList = []
        for i in range(vecNum):
            e = struct.unpack("f", f.read(4))[0]
            mEigenValList.append(e)
        mEigenVal = np.array(mEigenValList)

        mPrinCompList = []
        for i in range(vecNum):
            prinComp = []
            for j in range(ptNum):
                x,y,z = struct.unpack("fff", f.read(12))
                prinComp.append(np.array([x,y,z]))
            prinComp = np.stack(prinComp)
            mPrinCompList.append(prinComp)
        mPrinCompList = np.array(mPrinCompList)
        mPrinCompList = mPrinCompList.reshape(199,-1).transpose(1,0)
        print(mAvergPos.shape)
        print(mEigenVal.shape)
        print(mPrinCompList.shape)

        _, faces, _ = load_obj('./OriMesh.obj')
        faces = faces.verts_idx
        print(faces.shape)

    return {'shape_mean': mAvergPos, 'shape_pcaBasis': mPrinCompList, 'shape_pcaVariance': mEigenVal,
            'face': faces}


def load_depth():
    # Reading depth
    with open(depth_path, 'rb') as f:
        data = zlib.decompress(f.read(), -15)

    FRAME_COUNT = int(len(data) / w / h / 2)

    frames_in_meters = np.frombuffer(data, np.float16).reshape(FRAME_COUNT,h,w).copy()
    frames_in_meters = np.nan_to_num(frames_in_meters, 0)
    frames_in_mm = frames_in_meters * 1000.0

    # imgs_depth = (frames_in_mm - 100.0) / 400.0
    # imgs_depth = imgs_depth.clip(0.0, 1.0)
    # imgs_depth = (imgs_depth * 255.0).astype('uint8')
    mask = np.ones(frames_in_mm.shape)
    frames_in_mm[(frames_in_mm<100) + (frames_in_mm>450)] = 0
    mask[(frames_in_mm<100) + (frames_in_mm>450)] = 0
    imgs_depth = frames_in_mm.clip(100.0, 450.0)

    return imgs_depth, mask


def render_point(
    verts=None,
    rgb=None,
    image_size=(640,360),
    background_color=(1, 1, 1),
    device=None,
    output_path='pointcloud.gif',
    # R=None,
    # T=None
    # img=None,
    # mask=None
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    # point_cloud = np.load(point_cloud_path)
    # print(verts.shape)
    # y_c = (verts[:,1].max()+ verts[:,1].min())/2
    # verts = torch.Tensor(verts).to(device).unsqueeze(0)
    # print(verts.shape)
    if rgb==None:
        # rgb = torch.ones_like(verts)  # (1, N_v, 3)
        # rgb = rgb * torch.tensor([0, 0., 0], device=device)  # (1, N_v, 3)
        rgb = torch.zeros(verts.shape[1], verts.shape[2]+1).to(device).unsqueeze(0)
        rgb[:,:,3] =1
    else:
        rgb = torch.Tensor(rgb).to(device).unsqueeze(0)
    # print(rgb.shape)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    
    R = torch.eye(3).unsqueeze(0)
    # R = pytorch3d.transforms.euler_angles_to_matrix(
    #     torch.tensor([0, np.pi, 0]), "XYZ"
    # )
    # R = R.unsqueeze(0)
    T = torch.tensor([[0, 0, 0]])
    # print('R',R[0])
    # print('T', T[0])
    # cameras = pytorch3d.renderer.PerspectiveCameras(focal_length = ((436.16, 436.16), ), principal_point=((320.08, 179.22),), image_size=((640, 360),), R=R, T=T, device=device)
    cameras = pytorch3d.renderer.PerspectiveCameras(focal_length = ((436.16/180, 436.16/180), ), principal_point=((-(320.08-320)/320, -(179.22-180)/180.),), R=R, T=T, device=device)
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
    # rend = renderer(point_cloud.extend(num_views), cameras=cameras)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[:, ..., :3]#[:,::-1,:][:,:,::-1]  # (B, H, W, 4) -> (H, W, 3)
    # print(rend.shape)
    # plt.imsave('', image)
    # for i in range(12):
    #     print('rend'+str(i),rend[i].max())
    #     print('rend'+str(i),rend[i].min())
    # imageio.imsave('perspective.jpg', rend[0])

    # print(rend.shape)
    # plt.imsave('', image)
    # imageio.mimsave(output_path, list(rend), fps=5)
    return rend[0]


def render_mesh(
    verts=None,
    faces=None,
    rgb=None,
    image_size=(640,360),
    background_color=(1, 1, 1),
    device=None,
    output_path='pointcloud.gif',
    R=None,
    T=None
    # img=None,
    # mask=None
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_mesh_renderer(
        image_size=image_size
    )
    # point_cloud = np.load(point_cloud_path)
    # print(verts.shape)
    # y_c = (verts[:,1].max()+ verts[:,1].min())/2
    # verts = torch.Tensor(verts).to(device).unsqueeze(0)
    # print(verts.shape)
    # faces = torch.tensor(faces, device=device).transpose(0,1).unsqueeze(0)
    faces = faces.cuda().unsqueeze(0)
    # print(verts.shape)
    # print(faces.shape)
    # print('faces', faces.shape)
    if rgb==None:
        rgb = torch.ones_like(verts)  # (1, N_v, 3)
        rgb = rgb * torch.tensor([0.7, 0.7, 1.0], device=device)  # (1, N_v, 3)
    else:
        rgb = torch.Tensor(rgb).to(device).unsqueeze(0)
    # print(rgb.shape)
    # print(faces.min())
    # print(faces.max())
    mesh = pytorch3d.structures.Meshes(
        verts=verts,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(rgb),
    )
    R = torch.eye(3).unsqueeze(0)
    # R = pytorch3d.transforms.euler_angles_to_matrix(
    #     torch.tensor([0, np.pi, 0]), "XYZ"
    # )
    # R = R.unsqueeze(0)
    T = torch.tensor([[0, 0, 0]])

    lights = pytorch3d.renderer.PointLights(location=[[0, 0, 0]], device=device)
    # cameras = pytorch3d.renderer.PerspectiveCameras(focal_length = ((436.16, 436.16), ), principal_point=((320.08, 179.22),), image_size=((640, 360),), R=R, T=T, device=device)
    cameras = pytorch3d.renderer.PerspectiveCameras(focal_length = ((436.16/180, 436.16/180), ), principal_point=((-(320.08-320)/320, -(179.22-180)/180.),), R=R, T=T, device=device)
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
    # rend = renderer(point_cloud.extend(num_views), cameras=cameras)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[:, ..., :3]#[:,::-1,:][:,:,::-1]  # (B, H, W, 4) -> (H, W, 3)
    # print(rend.shape)
    # plt.imsave('', image)
    # for i in range(12):
    #     print('rend'+str(i),rend[i].max())
    #     print('rend'+str(i),rend[i].min())
    # imageio.imsave('mesh.jpg', rend[0])

    # print(rend.shape)
    # plt.imsave('', image)
    # imageio.mimsave(output_path, list(rend), fps=5)
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    fragments = rasterizer(mesh)
    z = fragments[1]
    return rend[0], z.cpu().numpy()


def fit_point(
    data=None,
    key_keypoint=None,
    ref_keypoint=None,
    ref_keypoint_side=None,
    image_size=(640,360),
    background_color=(0, 0, 0),
    device=None,
    num = 0,
    ref_depth=None
):
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )

    mean_shape = torch.tensor(data['shape_mean'], dtype=torch.float32).reshape(-1,3).to(device)
    pca_shape = torch.tensor(data['shape_pcaBasis'], dtype=torch.float32).to(device)
    variance_shape = torch.tensor(data['shape_pcaVariance'], dtype=torch.float32).to(device)

    model = Model(mean_shape, pca_shape, variance_shape, renderer, key_ref=ref_keypoint, key_side_ref=ref_keypoint_side, key_keypoint=key_keypoint, frame_num=num, ref_depth=ref_depth).to(device)
    optimizer = torch.optim.Adam([model.R, model.T], lr=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), 10.0)

    if USE_WANDB:
        wandb.init(project="fit-mesh")

    torch.cuda.empty_cache()

    epoch = tqdm(range(4000))
    R=None
    T=None
    print('stage 1')
    for i in epoch:
        optimizer.zero_grad()
        loss, loss1, reg_loss, loss_d, R, T, key, _, _ = model()
        l = loss + 0.1*loss1
        l.backward()
        optimizer.step()
        
        if i%1000==0:
            print('Optimizing (reg loss %.4f, loss %.4f)' % (reg_loss.data, loss.data))

        if USE_WANDB and i%1000==0:
            wandb.log({'epoch': i, 'train/loss': loss.data})

    epoch = tqdm(range(2000))
    optimizer = torch.optim.Adam([model.R, model.T, model.param], lr=0.005)
    
    print('stage 2')
    
    for i in epoch:
        flag=False
        if i%200 ==0:
            flag=True
        optimizer.zero_grad()
        if i>=1000 and i<1500:
            for batch in range(0, num, 120):
                loss, loss1, reg_loss, loss_d, R, T, key, m, _ = model(d=9000, b1=batch, b2=min(batch+120, num), update=flag)
                l = 0.3*loss+ 5*reg_loss + loss_d/5+ 0.3*0.1*loss1
                l.backward()
                optimizer.step()
        elif i>=1500 and i<1800:
            for batch in range(0, num, 120):
                loss, loss1, reg_loss, loss_d, R, T, key, m, _ = model(d=2500, b1=batch, b2=min(batch+120, num), update=flag)
                l = 0.3*loss+ 5*reg_loss + loss_d/5+ 0.3*0.1*loss1
                l.backward()
                optimizer.step()
        elif i>=1800:
            # if i%2==0 or i%2==1:
            for batch in range(0, num, 120):
                loss,loss1, reg_loss, loss_d, R, T, key, m, _ = model(d=400, b1=batch, b2=min(batch+120, num), update=flag)
                l = 0.3*loss+ 5*reg_loss + loss_d/3+ 0.3*0.1*loss1
                l.backward()
                optimizer.step()
            # else:
            #     loss,loss1, reg_loss, loss_d, R, T, key, m, _ = model()
            #     l = loss+ 10*reg_loss+ 0.1*loss1
            #     l.backward()
            #     optimizer.step()
        else:
            loss,loss1, reg_loss, loss_d, R, T, key, m, _ = model()
            l = loss+ 5*reg_loss + 0.1*loss1
            l.backward()
            optimizer.step()
        
        if i%100==0 and i>=1000:
            print('epoch %d, Optimizing (reg loss %.4f, keypoint loss %.4f, depth loss %.4f)' % (i, reg_loss.data, loss.data, loss_d.data))
            print(m.sum())
            print(m.sum(dim=(1,2)))
        elif i%500==0 and i<1000:
            print('epoch %d, Optimizing (reg loss %.4f, keypoint loss %.4f)' % (i, reg_loss.data, loss.data))
            # print(m.sum())

        if USE_WANDB and i%1000==0:
            wandb.log({'epoch': i, 'train/loss': loss.data})
        
    return model, R, T, model.param 

def write_pointcloud(filename,xyz_points,rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    # fid.write(bytes('property uchar red\n', 'utf-8'))
    # fid.write(bytes('property uchar green\n', 'utf-8'))
    # fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        # print(rgb_points[i,2].numpy().tostring())
        fid.write(bytearray(struct.pack("fff",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2])))
                                        # rgb_points[i,0].numpy().tobytes(),rgb_points[i,1].numpy().tobytes(),rgb_points[i,2].numpy().tobytes())))
    fid.close()


if __name__ == "__main__":

    device = get_device()

    with open("example_landmark_68_2019.anl","r") as f:
        loaded_list = f.read()
    loaded_list = loaded_list.split('\n')
    loaded_list = [int(i.split(',')[1]) for i in loaded_list]

    # with open("face.txt","r") as f:
    #     loaded_list = f.read()
    # loaded_list = loaded_list.split('\n')
    # loaded_list = [int(i) for i in loaded_list]
    # print(loaded_list)
    
    key_keypoint = torch.tensor(loaded_list)
    # key_keypoint = torch.tensor([16203, 16235, 16260, 16290, 27061, 22481, 22451, 22426, 22394, 8134, 8143, 8151, 8156, 6986, 7695, 8167, 8639, 9346, 2345, 4146, 5180, 6214, 4932, 4158, 10009, 11032, 12061, 13872, 12073, 11299, 5264, 6280, 7472, 8180, 8888, 10075, 11115, 9260, 8553, 8199, 7845, 7136, 7600, 8190, 8780, 8545, 8191, 7837, 4538, 11679])
    # print(key_keypoint)
    
    # print(key_keypoint.shape)
    # print(key_keypoint)

    cap = cv2.VideoCapture(video_path)
    # frameSize = (2 * 2160, 3840)
    # frameSize = (2 * 1080, 1920)
    # frameSize = (4 * 360, 640)
    frameSize = (4 * 360, 640)
    
    K = torch.tensor([[436.16, 0, 320.08], [0, 436.16, 179.22], [0, 0, 1]], dtype=torch.float)

    l = [122,131, 214, 217, 247, 253, 258, 262, 299, 352, 353, 367, 373, 619, 620, 850, 851, 852, 854, 874, 1018, 1020, 1024, 1025, 1026, 1041, 1068, 1138, 1196]
    num = 120
    print('num', num)
    frame_num = 0
    
    data = load_BFM_2019()
    # data = load_facewarehouse_bin()
    mask = torch.zeros((num,640,360)).cuda()
    video = np.zeros((num,640,360,3))
    imgs_depth, _ = load_depth()
    
    iter = 0
    # open_pose = torch.tensor([0, 1, 2, 3, 8, 13, 14, 15, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 65, 66, 67, 68, 69])
    # open_pose = torch.tensor([ 8, 30, 36,  39,  42,  45,  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67])
    open_pose = torch.tensor([8, 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67])
    open_pose_side = torch.tensor([0,1,2,3,4,5,6,7, 9,10,11,12,13,14,15,16])
    ref_keypoints = torch.zeros((num,open_pose.shape[0],2)).cuda()
    ref_keypoints_side = torch.zeros((num,open_pose_side.shape[0],2)).cuda()
    ref_depth = torch.zeros((num, 640, 360, 1), device = device, dtype=torch.float32)
   

    while cap.isOpened():
        ret, frame = cap.read()
        # print(frame_num)
        if ret and frame_num%10 ==0:
        # if ret: 
            # print('get in')
            cur_im = frame
            cur_im = cv2.transpose(frame)   
            cur_im = cv2.flip(cur_im, 1)

            cur_depth = np.flip(imgs_depth[frame_num,: ,:].T)
            cur_depth = cv2.flip(np.float32(cur_depth), -1)
            cur_depth = cv2.flip(cur_depth, 1)
            cur_depth = torch.tensor(cur_depth, device=device)
            # print(cur_depth.shape)
            ref_depth[iter,:,:,0] = cur_depth

            cur_im = cv2.resize(cur_im, (cur_depth.shape[1], cur_depth.shape[0]))
            video[iter] = cur_im

            with open('json/000001_%012d_keypoints.json'%(frame_num), 'r') as f:
                cur_keypoint = json.load(f)
                cur_keypoint = cur_keypoint['people'][0]['face_keypoints_2d']
                # cur_keypoint = [float(i) for i in cur_keypoint]
                cur_keypoint = torch.tensor(cur_keypoint)
                # print(cur_keypoint.reshape(-1,3).shape)
                cur_keypoint = cur_keypoint.reshape(-1,3)[:,:2]*360/1080
                # ref_keypoints[iter,:,1] = cur_keypoint[:, 0]
                # ref_keypoints[iter,:,0] = cur_keypoint[:, 1]

                cur_keypoint = cur_keypoint.long()
                mask[iter,cur_keypoint[:,1],cur_keypoint[:,0]] = 1
                ref_keypoints[iter,:,:] = cur_keypoint[open_pose,:2]
                ref_keypoints_side[iter,:,:] = cur_keypoint[open_pose_side,:2]

                
                # print('mask1',mask[iter])
                # for i in range(cur_keypoint.shape[0]):
                #     mask[iter][int(cur_keypoint[i][1])][int(cur_keypoint[i][0])] = 1
                # print('mask2',mask[iter])
                # print('delta', (mask1-mask2).sum())
                
            # verts, _ = unproject_depth_image(cur_im, cur_depth, K, mask[iter])


            iter+=1

            # cv2.imshow('frame', im_h)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_num = frame_num + 1
        if frame_num==1200:
            break
        print('%d' % frame_num, end="\r")
    print("")

    model, R, T, param = fit_point(data, key_keypoint, ref_keypoint=ref_keypoints, ref_keypoint_side=ref_keypoints_side, num=num, ref_depth = ref_depth)
    # pos = pos.detach()
    param = param.detach()

    _,_,_,_, R, T,  k, _, depth = model(d=0, b1=0, b2=min(50,num))
    k = k.detach().long().cpu().numpy()
    R = R.detach()
    T = T.detach()
    #------------------------
    
    dic = {'R':R, 'T':T, 'param': param}
    torch.save(dic, 'bfs.pth')

    
    # print(device)
    mean = torch.tensor(data['shape_mean'], dtype=torch.float32).reshape(-1,3).to(device)
    pca = torch.tensor(data['shape_pcaBasis'], dtype=torch.float32).to(device)
    variance = torch.tensor(data['shape_pcaVariance'], dtype=torch.float32).to(device)

    verts = mean + (pca @ (param * torch.sqrt(variance)).reshape(-1,1)).reshape(-1,3)
    # verts = mean
    # verts = verts*10
    verts = verts.unsqueeze(0)
    # R = pytorch3d.renderer.look_at_rotation(pos, device=device)
    # T = -torch.bmm(R.transpose(1, 2), pos[:, :, None])[:, :, 0]

    #------------------------
    # r = torch.eye(3).unsqueeze(0)
    # r = pytorch3d.transforms.euler_angles_to_matrix(
    #     torch.tensor([0, 0, np.pi/3]), "XYZ"
    # )
    # r = r.unsqueeze(0)
    # print(r)
    # t = torch.tensor([[0, 0, 300]])
    # mesh = render_mesh(verts=verts.detach().unsqueeze(0),faces=data['face'] , rgb=None, R = r, T=t)*255

    
    out = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc(*'avc1'), 30, frameSize)
    ref_keypoints = ref_keypoints.long().cpu().numpy()
    ref_keypoints_side = ref_keypoints_side.long().cpu().numpy()
    depth = depth.detach().cpu()#.numpy()
    ref_depth = ref_depth.cpu().numpy()
    for i in range(num):
        # print(R[i])
        # print(T[i])
        # gt_k = render_point(verts=ref_keypoints[i:i+1,:,:],rgb=None)*255
        # kk = render_point(verts=k[i:i+1,:,:],rgb=None)*255
        # if i%50==0:
        #     _,_,_,_, _, _,  _, _, depth = model(d=0, b1=i, b2=min(i+50,num))
        #     print(i)
        #     depth = depth.detach().cpu().numpy()
        gt_k = mask[i,:,:,None].broadcast_to((640,360,3)).cpu().numpy()*255
        # kk = k[i,:,:,:3].cpu().numpy()*255
        kk = np.zeros((640,360,3))
        
        kk[k[i, :, 1], k[i, :, 0], :] = 255
        
        # print('k', k[i])
        # print('ref_k', ref_keypoints[i])
        # print(gt_k.shape)
        # print(kk.shape)
        # kk = render_point(verts=verts[None, key_keypoint, :],rgb=None)*255
        
        # mesh = render_mesh(verts=verts.detach().unsqueeze(0),faces=data['face'] , rgb=None, R = R[i:i+1,: ,:], T=T[i:i+1,:])*255
        points = torch.bmm(verts, R[i:i+1]) + torch.broadcast_to(T[i:i+1].unsqueeze(1), (1, verts.shape[1], 3))
        mesh, z = render_mesh(verts=points.detach(),faces=data['face'] , rgb=None)
        mesh *= 255
        
        for j in range(52):
            video[i] = cv2.circle(video[i], (ref_keypoints[i,j,0], ref_keypoints[i,j,1]), 3, (255, 0, 0), -1)
            
        for j in range(16):
            video[i] = cv2.circle(video[i], (ref_keypoints_side[i,j,0], ref_keypoints_side[i,j,1]), 3, (255, 0, 0), -1)
            
        for j in range(68):
            video[i] = cv2.circle(video[i], (k[i,j,0], k[i,j,1]), 3, (0, 0, 255), -1)
            # mesh = cv2.circle(mesh, (k[i,j,0], k[i,j,1]), 3, (0, 0, 255), -1)

        # depth_i = cv2.applyColorMap(((depth[i]-50)/450*255).astype('uint8'), cv2.COLORMAP_JET)#.transpose(1,0,2)
        # if i==0:
        #     print('pred', depth[i%20, 315:325, 175:185, :])
        #     print('ref', ref_depth[i, 315:325, 175:185, :])
        # depth_i = (depth[i%50]/450*255).broadcast_to((640,360,3)).numpy().astype('uint8')
        z[0] = z[0].clip(100.0, 450.0)
        depth_i = cv2.applyColorMap((z[0]/450*255).astype('uint8'), cv2.COLORMAP_JET)#.transpose(1,0,2)
        # print(depth_i.shape)
        # ref_depth_i = cv2.applyColorMap(((ref_depth[i]-50)/450*255).astype('uint8'), cv2.COLORMAP_JET)#.transpose(1,0,2)
        # print('gt depth', (ref_depth[i]-50)/450*255)
        # ref_depth_i = (ref_depth[i]/450*255).broadcast_to((640,360,3)).numpy().astype('uint8')
        ref_depth_i = cv2.applyColorMap((ref_depth[i]/450*255).astype('uint8'), cv2.COLORMAP_JET)#.transpose(1,0,2)
        # print(ref_depth_i.shape)

        # print(video[i].shape)
        # print(mesh.shape)
        # im_h = cv2.hconcat([video[i].astype('uint8'), kk.astype('uint8'),gt_k.astype('uint8'),mesh.astype('uint8')])
        im_h = cv2.hconcat([video[i].astype('uint8'),mesh.astype('uint8'), depth_i.astype('uint8'), ref_depth_i.astype('uint8')])
        out.write(im_h)

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    
