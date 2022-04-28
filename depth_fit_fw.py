import json
from weakref import ref
import pandas as pd
import h5py

import cv2
import os
import os.path as path

import imageio
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np

from PIL import Image, ImageDraw
from tqdm.auto import tqdm

import pickle as pkl

import zlib
import struct
import torch.nn as nn

import wandb
USE_WANDB = False
import torch.nn.functional as F

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

from utils import (
    get_device, 
    get_mesh_renderer, 
    get_points_renderer, 
    unproject_depth_image
)

from model import Model

model_bin_path = "face_warehouse/FaceMorphModel_199.bin"
obj_path = "face_warehouse/OriMesh.obj"
face_keypoints_mapping_path = "face_warehouse/face.txt"
leftear_keypoints_mapping_path = "face_warehouse/left_ear.txt"
rightear_keypoints_mapping_path = "face_warehouse/right_ear.txt"

face_keypoints_dir = "data/face_keypoints"
ear_keypoints_path = "data/ear_keypoints_prediction.pkl"

w = 640
h = 360
data_folder = 'data'
depth_path = path.join(data_folder ,'depth.xyz')
video_path = path.join(data_folder , 'rgb.mov')
output_video_path = path.join(data_folder , '/rgbd.mp4')

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

def load_facewarehouse_bin(bin_path, obj_path):
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

        _, faces, _ = load_obj(obj_path)
        faces = faces.verts_idx

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
    if rgb==None:
        
        rgb = torch.zeros(verts.shape[1], verts.shape[2]+1).to(device).unsqueeze(0)
        rgb[:,:,3] =1
    else:
        rgb = torch.Tensor(rgb).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    
    R = torch.eye(3).unsqueeze(0)
    
    T = torch.tensor([[0, 0, 0]])
    
    cameras = pytorch3d.renderer.PerspectiveCameras(focal_length = ((436.16/180, 436.16/180), ), principal_point=((-(320.08-320)/320, -(179.22-180)/180.),), R=R, T=T, device=device)
    
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[:, ..., :3]#[:,::-1,:][:,:,::-1]  # (B, H, W, 4) -> (H, W, 3)
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
    
    faces = faces.cuda().unsqueeze(0)
    
    if rgb==None:
        rgb = torch.ones_like(verts)  # (1, N_v, 3)
        rgb = rgb * torch.tensor([0.7, 0.7, 1.0], device=device)  # (1, N_v, 3)
    else:
        rgb = torch.Tensor(rgb).to(device).unsqueeze(0)
    
    mesh = pytorch3d.structures.Meshes(
        verts=verts,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(rgb),
    )
    R = torch.eye(3).unsqueeze(0)
    
    T = torch.tensor([[0, 0, 0]])

    lights = pytorch3d.renderer.PointLights(location=[[0, 0, 0]], device=device)
    cameras = pytorch3d.renderer.PerspectiveCameras(focal_length = ((436.16/180, 436.16/180), ), principal_point=((-(320.08-320)/320, -(179.22-180)/180.),), R=R, T=T, device=device)
    
    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[:, ..., :3]#[:,::-1,:][:,:,::-1]  # (B, H, W, 4) -> (H, W, 3)
    
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

epoch_1 = 4000
#epoch_1 = 200
epoch_2 = 2500
#epoch_2 = 200
def fit_point(
    data=None, 
    face_keypoints_indices=None, 
    ref_face_keypoints=None, 
    ref_face_keypoints_side=None, 
    ref_face_keypoints_score=None, 
    ref_face_keypoints_side_score=None, 
    leftear_keypoints_indices=None,
    ref_leftear_keypoints=None,
    ref_leftear_keypoints_score=None,
    rightear_keypoints_indices=None,
    ref_rightear_keypoints=None,
    ref_rightear_keypoints_score=None,
    image_size=(640,360),
    background_color=(0, 0, 0),
    device=None,
    frame_total = 0,
    ref_depth=None
):
    if device is None:
        device = get_device()

    renderer = get_points_renderer(image_size=image_size, background_color=background_color)

    mean_shape = torch.tensor(data['shape_mean'], dtype=torch.float32).reshape(-1,3).to(device)
    pca_shape = torch.tensor(data['shape_pcaBasis'], dtype=torch.float32).to(device)
    variance_shape = torch.tensor(data['shape_pcaVariance'], dtype=torch.float32).to(device)

    model = Model(
        mean = mean_shape, 
        pca = pca_shape, 
        variance = variance_shape, 
        renderer = renderer, 

        face_keypoints_indices=face_keypoints_indices, 
        ref_face_keypoints=ref_face_keypoints, 
        ref_face_keypoints_side=ref_face_keypoints_side, 
        ref_face_keypoints_score=ref_face_keypoints_score, 
        ref_face_keypoints_side_score=ref_face_keypoints_side_score, 

        leftear_keypoints_indices=leftear_keypoints_indices,
        ref_leftear_keypoints=ref_leftear_keypoints,
        ref_leftear_keypoints_score=ref_leftear_keypoints_score,

        rightear_keypoints_indices=rightear_keypoints_indices,
        ref_rightear_keypoints=ref_rightear_keypoints,
        ref_rightear_keypoints_score=ref_rightear_keypoints_score,
       
        image_size=image_size, 
        device=device,
        frame_total=frame_total, 
        ref_depth=ref_depth)
    model = model.to(device)

    optimizer = torch.optim.Adam([model.R, model.T], lr=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), 10.0)

    if USE_WANDB:
        wandb.init(project="fit-mesh")

    torch.cuda.empty_cache()

    epoch = tqdm(range(epoch_1))
    R=None
    T=None
    print('stage 1')
    for i in epoch:
        flag=False
        if i%100 ==0:
            flag=True
        if i<4000:
            optimizer.zero_grad()
            loss, loss1, leftear_loss, rightear_loss, reg_loss, loss_d, R, T, key, leftear_kp, rightear_kp, _, _ = model()
            #l = loss + 0.1*loss1
            l = loss + leftear_loss + rightear_loss + 0.1*loss1
            l.backward()
            optimizer.step()
        else:
            loss, loss1, leftear_loss, rightear_loss, reg_loss, loss_d, R, T, key, leftear_kp, rightear_kp, m, _ = model(d=900, b1=0, b2=120, update=flag)
            #l = loss + 0.1*loss1 +loss_d/50
            l = loss + leftear_loss + rightear_loss + 0.1*loss1 +loss_d/50

            l.backward()
            optimizer.step()
        
        if i<4000 and i%1000==0:
            print('Optimizing (reg loss %.4f, loss %.4f)' % (reg_loss.data, loss.data))
        elif i%100==0 and i>=4000:
            print('epoch %d, Optimizing (reg loss %.4f, keypoint loss %.4f, depth loss %.4f)' % (i, reg_loss.data, loss.data, loss_d.data))
            print(m.sum())
            print(m.sum(dim=(1,2)))

        if USE_WANDB and i%1000==0:
            wandb.log({'epoch': i, 'train/loss': loss.data})

    epoch = tqdm(range(epoch_2))
    optimizer = torch.optim.Adam([model.R, model.T, model.param], lr=0.01)
    
    print('stage 2')
    
    for i in epoch:
        flag=False
        if i%100 ==0:
            flag=True
        optimizer.zero_grad()
        if i>=1000 and i<1500:
            for batch in range(0, frame_total, 120):
                loss, loss1, leftear_loss, rightear_loss, reg_loss, loss_d, R, T, key, leftear_kp, rightear_kp, m, _ = model(d=900, b1=batch, b2=min(batch+120, frame_total), update=flag)
                #l = 0.3*loss+ 5*reg_loss + loss_d/3+ 0.3*0.5*loss1
                l = 0.3* ( loss + leftear_loss + rightear_loss)+ 5*reg_loss + loss_d/3+ 0.3*0.5*loss1
                l.backward()
                optimizer.step()
        elif i>=1500 and i<2100:
            for batch in range(0, frame_total, 120):
                loss, loss1, leftear_loss, rightear_loss, reg_loss, loss_d, R, T, key, leftear_kp, rightear_kp, m, _= model(d=400, b1=batch, b2=min(batch+120, frame_total), update=flag)
                #l = 0.3*loss+ 5*reg_loss + loss_d/3+ 0.3*0.5*loss1
                l = 0.3* ( loss + leftear_loss + rightear_loss )+ 5*reg_loss + loss_d/3+ 0.3*0.5*loss1
                l.backward()
                optimizer.step()
        elif i>=2100:
            # if i%2==0 or i%2==1:
            for batch in range(0, frame_total, 120):
                loss, loss1, leftear_loss, rightear_loss, reg_loss, loss_d, R, T, key, leftear_kp, rightear_kp, m, _= model(d=100, b1=batch, b2=min(batch+120, frame_total), update=flag)
                #l = 0.3*loss+ 5*reg_loss + loss_d/3+ 0.3*0.5*loss1
                l = 0.3* ( loss + leftear_loss + rightear_loss )+ 5*reg_loss + loss_d/3+ 0.3*0.5*loss1
                l.backward()
                optimizer.step()
            # else:
            #     loss,loss1, reg_loss, loss_d, R, T, key, m, _ = model()
            #     l = loss+ 10*reg_loss+ 0.1*loss1
            #     l.backward()
            #     optimizer.step()
        else:
            loss, loss1, leftear_loss, rightear_loss, reg_loss, loss_d, R, T, key, leftear_kp, rightear_kp, m, _= model()
            #l = loss+ 5*reg_loss + 0.1*loss1
            l = loss + leftear_loss + rightear_loss + 5*reg_loss + 0.1*loss1
            l.backward()
            optimizer.step()
        
        if i>=1000 and i%100==0:
            print('epoch %d, Optimizing (reg loss %.4f, keypoint loss %.4f, depth loss %.4f)' % (i, reg_loss.data, loss.data, loss_d.data))
            print(m.sum())
            print(m.sum(dim=(1,2)))
        elif i%500==0 and i<1000:
            print('epoch %d, Optimizing (reg loss %.4f, keypoint loss %.4f)' % (i, reg_loss.data, loss.data))

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
    
    width = 360
    height = 640
    original_width = 1080
    original_height = 1920
    device = get_device()

    with open(face_keypoints_mapping_path,"r") as f:
        face_keypoints_mapping_list = f.read()
    face_keypoints_mapping_list = face_keypoints_mapping_list.strip().split('\n')
    face_keypoints_mapping_list = [int(i) for i in face_keypoints_mapping_list]

    with open(leftear_keypoints_mapping_path,"r") as f:
        leftear_keypoints_mapping_list = f.read()
    leftear_keypoints_mapping_list = leftear_keypoints_mapping_list.strip().split('\n')
    leftear_keypoints_mapping_list = [int(i) for i in leftear_keypoints_mapping_list]

    with open(rightear_keypoints_mapping_path,"r") as f:
        rightear_keypoints_mapping_list = f.read()
    rightear_keypoints_mapping_list = rightear_keypoints_mapping_list.strip().split('\n')
    rightear_keypoints_mapping_list = [int(i) for i in rightear_keypoints_mapping_list]
    
    face_keypoints_indices = torch.tensor(face_keypoints_mapping_list)
    leftear_keypoints_indices = torch.tensor(leftear_keypoints_mapping_list)
    rightear_keypoints_indices = torch.tensor(rightear_keypoints_mapping_list)

    cap = cv2.VideoCapture(video_path)
    frameSize = (5 * width, height)
    
    K = torch.tensor([[436.16, 0, 320.08], [0, 436.16, 179.22], [0, 0, 1]], dtype=torch.float)
    l = [122,131, 214, 217, 247, 253, 258, 262, 299, 352, 353, 367, 373, 619, 620, 850, 851, 852, 854, 874, 1018, 1020, 1024, 1025, 1026, 1041, 1068, 1138, 1196]
    frame_total = 120
    frame_num = 0
    
    # data = load_BFM_2019()
    data = load_facewarehouse_bin(model_bin_path, obj_path)
    mask = torch.zeros((frame_total,height,width)).to(device)
    video = np.zeros((frame_total,height,width,3))
    imgs_depth, _ = load_depth()
    
    iter = 0
    # face_keypoints_order = torch.tensor([0, 1, 2, 3, 8, 13, 14, 15, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 65, 66, 67, 68, 69])
    # face_keypoints_order = torch.tensor([ 8, 30, 36,  39,  42,  45,  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67])
    face_keypoints_order = torch.tensor([8, 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67])
    side_face_keypoints_order = torch.tensor([0,1,2,3,4,5,6,7, 9,10,11,12,13,14,15,16])
    
    nose_tip_indice = 30

    face_keypoints_total = face_keypoints_order.shape[0]
    side_face_keypoints_total = side_face_keypoints_order.shape[0]
    leftear_keypoints_total = leftear_keypoints_indices.shape[0]
    rightear_keypoints_total = rightear_keypoints_indices.shape[0]
    
    ref_face_keypoints = torch.zeros((frame_total,face_keypoints_total,2)).to(device)
    ref_face_keypoints_side = torch.zeros((frame_total,side_face_keypoints_total,2)).to(device)
    ref_face_keypoints_score = torch.zeros((frame_total,face_keypoints_total,1)).to(device)
    ref_face_keypoints_side_score = torch.zeros((frame_total,side_face_keypoints_total,1)).to(device)

    ref_leftear_keypoints = torch.zeros([frame_total, leftear_keypoints_total, 2]).to(device)
    ref_leftear_keypoints_score = torch.zeros([frame_total, leftear_keypoints_total, 1]).to(device)

    ref_rightear_keypoints = torch.zeros([frame_total, rightear_keypoints_total, 2]).to(device)
    ref_rightear_keypoints_score = torch.zeros([frame_total, rightear_keypoints_total, 1]).to(device)

    ref_depth = torch.zeros((frame_total, height, width, 1), device = device, dtype=torch.float32)

    frame_sample_freq = 10

    ear_keypoints = pkl.load(open(ear_keypoints_path, 'rb'))

    # keypoint format [ w, h, c]

    while cap.isOpened():
        ret, frame = cap.read()
        if ret and frame_num % frame_sample_freq == 0:
        
            cur_im = frame
            cur_im = cv2.transpose(frame)   
            cur_im = cv2.flip(cur_im, 1)

            cur_depth = np.flip(imgs_depth[frame_num,: ,:].T)
            cur_depth = cv2.flip(np.float32(cur_depth), -1)
            cur_depth = cv2.flip(cur_depth, 1)
            cur_depth = torch.tensor(cur_depth, device=device)
            ref_depth[iter,:,:,0] = cur_depth

            cur_im = cv2.resize(cur_im, (cur_depth.shape[1], cur_depth.shape[0]))
            video[iter] = cur_im

            face_keypoints_path = path.join(face_keypoints_dir, '000001_%012d_keypoints.json'%(frame_num)) 
            with open(face_keypoints_path,'r') as f:
                cur_keypoint = json.load(f)
                cur_keypoint = cur_keypoint['people'][0]['face_keypoints_2d']
                
                cur_keypoint = torch.tensor(cur_keypoint)
                cur_keypoint = cur_keypoint.reshape(-1,3)

                ref_face_keypoints_score[iter,:,:] = cur_keypoint[face_keypoints_order,2:3]
                ref_face_keypoints_side_score[iter,:,:] = cur_keypoint[side_face_keypoints_order,2:3]

                cur_keypoint = (cur_keypoint[:,:2]*width/original_width).long()
                mask[iter,cur_keypoint[:,1],cur_keypoint[:,0]] = 1
                ref_face_keypoints[iter,:,:] = cur_keypoint[face_keypoints_order,:2]
                ref_face_keypoints_side[iter,:,:] = cur_keypoint[side_face_keypoints_order,:2]
            
                nose_tip_position = cur_keypoint[nose_tip_indice, :2] 

            ear_key = "image%04d.png"%(frame_num+1) 
            try:
                ear_data = ear_keypoints[ear_key]
                ear_kp = torch.tensor(ear_data[:2, :])
                ear_kp = (ear_kp * width / original_width)
                ear_mean_position = ear_kp.mean(axis=1)
                ear_kp = ear_kp.long()
                ear_kp = torch.transpose(ear_kp, 0, 1)
                ear_score = torch.tensor(ear_data[2, :]).unsqueeze(dim=1)
                
                if ear_mean_position[0] > nose_tip_position[0]:
                    # left ear
                    ref_leftear_keypoints[iter, :, :] = ear_kp
                    ref_leftear_keypoints_score[iter, :, :] = ear_score 

                    ref_rightear_keypoints[iter, :, :] = torch.zeros(ear_kp.shape)
                    ref_rightear_keypoints_score[iter, :, :] = torch.zeros(ear_score.shape) 
                else:
                    # righ ear
                    ref_leftear_keypoints[iter, :, :] = torch.zeros(ear_kp.shape)
                    ref_leftear_keypoints_score[iter, :, :] = torch.zeros(ear_score.shape)
                                                                
                    ref_rightear_keypoints[iter, :, :] = ear_kp
                    ref_rightear_keypoints_score[iter, :, :] = ear_score 

                # TODO : mask ? 
            except:
                ref_leftear_keypoints[iter, :, :] = torch.zeros([leftear_keypoints_total, 2])
                ref_leftear_keypoints_score[iter, :, :] = torch.zeros([leftear_keypoints_total, 1])
                ref_rightear_keypoints[iter, :, :] = torch.zeros([rightear_keypoints_total, 2])
                ref_rightear_keypoints_score[iter, :, :] = torch.zeros([rightear_keypoints_total, 1])
                pass 

            iter+=1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_num = frame_num + 1
        if frame_num >= frame_sample_freq * frame_total:
            break
        print('%d' % frame_num, end="\r")
    print("")

    model, R, T, param = fit_point(
        data, 
        face_keypoints_indices, 
        ref_face_keypoints=ref_face_keypoints, 
        ref_face_keypoints_side=ref_face_keypoints_side, 
        ref_face_keypoints_score=ref_face_keypoints_score, 
        ref_face_keypoints_side_score=ref_face_keypoints_side_score, 

        leftear_keypoints_indices=leftear_keypoints_indices,
        ref_leftear_keypoints=ref_leftear_keypoints,
        ref_leftear_keypoints_score=ref_leftear_keypoints_score,

        rightear_keypoints_indices=rightear_keypoints_indices,
        ref_rightear_keypoints=ref_rightear_keypoints,
        ref_rightear_keypoints_score=ref_rightear_keypoints_score,

        device=device,
        image_size= (height, width),
        frame_total=frame_total, 
        ref_depth = ref_depth
    )
    # pos = pos.detach()
    param = param.detach()

    _,_,_,_,_,_,R,T,k,leftear_k,rightear_k,_,_= model()
    k = k.detach().long().cpu().numpy()
    R = R.detach()
    T = T.detach()
    
    dic = {'R':R, 'T':T, 'param': param}
    torch.save(dic, 'huber1.pth')

    mean = torch.tensor(data['shape_mean'], dtype=torch.float32).reshape(-1,3).to(device)
    pca = torch.tensor(data['shape_pcaBasis'], dtype=torch.float32).to(device)
    variance = torch.tensor(data['shape_pcaVariance'], dtype=torch.float32).to(device)

    verts = mean + (pca @ (param * torch.sqrt(variance)).reshape(-1,1)).reshape(-1,3)
    verts = verts.unsqueeze(0)
    
    #out = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc(*'avc1'), 30, frameSize)
    out = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc(*'mp4v'), 30, frameSize)

    ref_face_keypoints = ref_face_keypoints.long().cpu().numpy()
    ref_face_keypoints_side = ref_face_keypoints_side.long().cpu().numpy()

    ref_leftear_keypoints = ref_leftear_keypoints.long().cpu().numpy()
    ref_rightear_keypoints = ref_rightear_keypoints.long().cpu().numpy()

    ref_depth = ref_depth.cpu().numpy()
    from IPython import embed;embed()
    for i in range(frame_total):
        gt_k = mask[i,:,:,None].broadcast_to((height,width,3)).cpu().numpy()*255
        
        kk = np.zeros((height,width,3))
        kk[k[i, :, 1], k[i, :, 0], :] = 255

        points = torch.bmm(verts, R[i:i+1]) + torch.broadcast_to(T[i:i+1].unsqueeze(1), (1, verts.shape[1], 3))
        mesh, z = render_mesh(verts=points.detach(),faces=data['face'] , rgb=None)
        mesh *= 255
        
        for j in range(face_keypoints_total - side_face_keypoints_total):
            video[i] = cv2.circle(video[i], (ref_face_keypoints[i,j,0], ref_face_keypoints[i,j,1]), 3, (255, 0, 0), -1)
            
        for j in range(side_face_keypoints_total):
            video[i] = cv2.circle(video[i], (ref_face_keypoints_side[i,j,0], ref_face_keypoints_side[i,j,1]), 3, (255, 0, 0), -1)
            
        for j in range(face_keypoitns_total):
            video[i] = cv2.circle(video[i], (k[i,j,0], k[i,j,1]), 3, (0, 0, 255), -1)
        
        z[0] = z[0].clip(100.0, 450.0)
        depth_i = cv2.applyColorMap((z[0]/450*255).astype('uint8'), cv2.COLORMAP_JET)#.transpose(1,0,2)
        
        ref_depth_i = cv2.applyColorMap((ref_depth[i]/450*255).astype('uint8'), cv2.COLORMAP_JET)#.transpose(1,0,2)
        dif_depth = np.absolute(ref_depth[i]-z[0]).clip(0, 100)
        dif_depth = cv2.applyColorMap((dif_depth/100*255).astype('uint8'), cv2.COLORMAP_JET)
        
        im_h = cv2.hconcat([video[i].astype('uint8'),mesh.astype('uint8'), depth_i.astype('uint8'), ref_depth_i.astype('uint8'), dif_depth.astype('uint8')])
        out.write(im_h)

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    
