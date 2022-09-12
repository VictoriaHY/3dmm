import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d
from tqdm import tqdm

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

import numpy as np
import math

class Model(nn.Module):
    def __init__(
        self, 
        mm_model, 
        renderer, 

        keypoints_dict,
        
        image_size=None,
        device=None,
        frame_total=None, 
        ref_depth=None,
        ref_contour=None):
        super().__init__()

        self.mean = mm_model["mean"]
        self.pca = mm_model["pca"]
        self.variance = mm_model["variance"]

        self.face_keypoints_indices = keypoints_dict["indice"]["face"]
        self.ref_depth = ref_depth
        self.ref_contour = ref_contour
        
        self.inner = torch.tensor([8, 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67])
        self.side = torch.tensor([0,1,2,3,4,5,6,7, 9,10,11,12,13,14,15,16])
        
        self.batchsize = frame_total
        
        self.device = device
        self.renderer = renderer
        
        self.ref_face_keypoints_inner = keypoints_dict["keypoints"]["face_inner"]
        self.ref_face_keypoints_side = keypoints_dict["keypoints"]["face_side"]
        self.ref_face_keypoints_inner_score = keypoints_dict["score"]["face_inner"]
        self.ref_face_keypoints_side_score = keypoints_dict["score"]["face_side"]

        self.leftear_keypoints_indices = keypoints_dict["indice"]["leftear"]
        self.ref_leftear_keypoints = keypoints_dict["keypoints"]["leftear"]
        self.ref_leftear_keypoints_score = keypoints_dict["score"]["leftear"]

        self.rightear_keypoints_indices = keypoints_dict["indice"]["rightear"]
        self.ref_rightear_keypoints = keypoints_dict["keypoints"]["rightear"]
        self.ref_rightear_keypoints_score = keypoints_dict["score"]["rightear"]
        
        R = torch.zeros((frame_total, 3), dtype=torch.float32, device=self.device)
        R[:,2] = np.pi
        R[:,1] = np.pi
        self.R = nn.Parameter(R)
        T = torch.zeros((frame_total, 3), dtype=torch.float32, device=self.device)
        T[:,2] = 300
        self.T = nn.Parameter(T)

        self.param = nn.Parameter(torch.zeros(self.pca.shape[1], dtype=torch.float32, device=self.device))
        # self.param = 0
        self.criterion = torch.nn.MSELoss()

        self.camera = pytorch3d.renderer.PerspectiveCameras(focal_length = ((436.16/180, 436.16/180), ), principal_point=((-(320.08-320)/320, -(179.22-180)/180.),), device=self.device)
        p_matrix = np.array([436.16, 0.0, 179.22, \
                            0.0, 436.16, 320.08, \
                            0.0, 0.0, 1.0], dtype=np.float32)

        p_matrix = np.tile(p_matrix.reshape(1, 3, 3), [self.batchsize, 1, 1])
        reverse_z = np.tile(np.reshape(np.array([1.0,0,0,0,1,0,0,0,-1.0], dtype=np.float32),[1,3,3]),[self.batchsize,1,1])
        
        self.p_matrix = torch.tensor(p_matrix, device=self.device)
        self.reverse_z = torch.tensor(reverse_z, device=self.device)

        # self.vertex_total = 53149
        self.vertex_total = 11510

        self.face_keypoints_total = self.face_keypoints_indices.shape[0]
        self.leftear_keypoints_total = self.leftear_keypoints_indices.shape[0]
        self.rightear_keypoints_total = self.rightear_keypoints_indices.shape[0]
       
        # [ B x H x W x 2] 
        self.corresp = torch.ones((frame_total, image_size[0], image_size[1], 2), device=self.device)
        self.m = (self.corresp>0)

        # [ B x H x W ]
        #self.vertex_image = torch.ones((frame_total, image_size[0], image_size[1]), device=self.device)
        # [ H x W ]
        self.image_size = image_size

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

        # DEBUG : camera_pos all zero ?  
        camera_pos = torch.tensor([0.0,0.0,0.0], device=self.device).reshape(1, 1, 3)

        points = torch.matmul(points, self.reverse_z[:batchsize,:,:]) + camera_pos
        aug_projection = torch.matmul(points,self.p_matrix[:batchsize,:,:].permute((0,2,1)))

        face_projection = aug_projection[:,:,:]/torch.reshape(aug_projection[:,:,2],[batchsize,-1,1])
        return face_projection
    
    def forward(self, d=0, b1=0, b2=0, update=False):
        
        # Render the image using the updated camera position. 
        # Based on the new position of the camera we calculate the rotation and translation matrices

        R1 = self.Compute_rotation_matrix(self.R)
        T1 = self.T

        points = self.mean + (self.pca @ (self.param * torch.sqrt(self.variance)).reshape(-1,1)).reshape(-1,3)
        points = points.unsqueeze(0)

        face_kp = points[:, self.face_keypoints_indices, :]
        face_kp = torch.bmm(torch.broadcast_to(face_kp, (self.batchsize,self.face_keypoints_total, 3)), R1)
        face_kp += torch.broadcast_to(T1.unsqueeze(1), (self.batchsize,self.face_keypoints_total, 3))
        face_kp = self.project(face_kp, self.batchsize)
        face_kp = face_kp[:,:,:2]

        leftear_kp = points[:, self.leftear_keypoints_indices, :]
        leftear_kp = torch.bmm(torch.broadcast_to(leftear_kp, (R1.shape[0], self.leftear_keypoints_total, 3)), R1)
        leftear_kp += torch.broadcast_to(T1.unsqueeze(1), (T1.shape[0], self.leftear_keypoints_total, 3))
        leftear_kp = self.project(leftear_kp, self.batchsize)
        leftear_kp = leftear_kp[:, :, :2]

        rightear_kp = points[:, self.rightear_keypoints_indices, :]
        rightear_kp = torch.bmm(torch.broadcast_to(rightear_kp, (R1.shape[0], self.rightear_keypoints_total, 3)), R1)
        rightear_kp += torch.broadcast_to(T1.unsqueeze(1), (T1.shape[0], self.rightear_keypoints_total, 3))
        rightear_kp = self.project(rightear_kp, self.batchsize)
        rightear_kp = rightear_kp[:, :, :2]

        kp_loss_face_inner = torch.mean(((face_kp[:, self.inner, :] - self.ref_face_keypoints_inner) ** 2) * self.ref_face_keypoints_inner_score)
        kp_loss_face_side = torch.mean(((face_kp[:, self.side, :] - self.ref_face_keypoints_side) ** 2) * self.ref_face_keypoints_side_score)

        kp_loss_leftear = torch.mean(((leftear_kp - self.ref_leftear_keypoints) ** 2) * self.ref_leftear_keypoints_score)
        kp_loss_rightear = torch.mean(((rightear_kp - self.ref_rightear_keypoints) ** 2) * self.ref_rightear_keypoints_score)

        loss_reg = torch.square(self.param).mean()
        loss_d = None
        loss_d1=None
        mask=None
        pred_d = None

        loss_seg = 0

        mesh_seg_list = [torch.empty([0,0])] * (b2-b1)
        if b2>0:
            
            points = torch.bmm(torch.broadcast_to(points, (b2-b1,self.vertex_total, 3)), R1[b1:b2,:,:]) + torch.broadcast_to(T1[b1:b2,:].unsqueeze(1), (b2-b1,self.vertex_total, 3))
            points1 = self.project(points, b2-b1)

            m0 = ((points1[:,:,1]>=0) & (points1[:,:,1]<self.image_size[0])) & ((points1[:,:,0]>=0) & (points1[:,:,0]<self.image_size[1]))
            m0 = m0[:,:,None].detach()
            points1 = points1 * m0
            
            if update:
                self.corresp[:,:,:,1] = 1       
                for i in tqdm(range(b1, b2)):
                    for j in range(self.vertex_total):
                        self.corresp[i, points1[i-b1,j,1].long(), points1[i-b1,j,0].long(),1] = j/(self.vertex_total)*2-1
                self.corresp.detach()
                
                print("calculating correspondence end")
            '''
            #has_vertex = (self.corresp[:,:,:,1]>=-1) & (self.corresp[:,:,:,1]<1)
            has_vertex = (self.corresp[:,:,:,1]>-1)
            has_vertex = has_vertex.float().cpu().numpy()
            #from IPython import embed;embed()

            sobel_kernel_size = 3

            edges = cv2.Sobel(has_vertex, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=sobel_kernel_size)

            B,H,W = np.where(edges!=0)
            B = torch.tensor(B)
            H = torch.tensor(H) 
            W = torch.tensor(W) 
            
            mesh_seg_list = []
            for i in range(b1, b2):
                bind = torch.where(B == i)
                # [ M x 2 ]
                mesh_seg = torch.stack((W[bind], H[bind]), dim=1) 
                mesh_seg_list.append(mesh_seg)

            H = H / self.image_size[0]
            W = W / self.image_size[1]

            for i in range(b1, b2):
                bind = torch.where(B == i)
                # [ M x 2 ]
                mesh_seg = torch.stack((W[bind], H[bind]), dim=1) 
                ref_seg = torch.tensor(self.ref_contour[i]).long()

                M = mesh_seg.shape[0]
                N = ref_seg.shape[0]
                
                # [M x N x 2]
                mesh_seg_rep = mesh_seg.repeat([N,1,1]).transpose(0,1)
                ref_seg_rep = ref_seg.repeat([M,1,1])
                dis = ((mesh_seg_rep - ref_seg_rep) ** 2).sum(dim=2).min(dim=1).values.sum()
                print(i, dis)
                loss_seg += dis

            self.corresp = self.corresp/(self.vertex_total-1)*2-1
            '''
            self.m = (self.corresp>=-1) & (self.corresp<1)
            
            points[:,self.vertex_total-1,2]=0
            pred_d = F.grid_sample(points[:,None,:,:], self.corresp[b1:b2, :,:,:], mode='nearest', align_corners=True)
            
            pred_d = pred_d#*self.m[b1:b2,None, : , :, 1]
            pred_d = pred_d.squeeze(1)
            
            ref = self.ref_depth[b1:b2,:,:,0]
            loss_d = ((pred_d - ref)**2)
            
            mask = (loss_d<d) & (loss_d>1e-6) & (pred_d>0)
            mask = mask.detach()
            
            loss_d1 = torch.nn.HuberLoss(delta=math.sqrt(d)/2)(pred_d[mask], ref[mask])
           
        loss_kp = {
            "face_inner": kp_loss_face_inner,
            "face_side": kp_loss_face_side,
            "leftear": kp_loss_leftear,
            "rightear": kp_loss_rightear} 

        loss = {
            "keypoints": loss_kp,
            "reg": loss_reg,
            "depth": loss_d1,
            "segmentation": loss_seg
            }

        transformation = {"R":R1, "T": T1}
        keypoints = {
            "face": face_kp,
            "leftear": leftear_kp,
            "rightear": rightear_kp
            }

        return loss, transformation, keypoints, mask, pred_d, mesh_seg_list

