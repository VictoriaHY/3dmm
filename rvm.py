import cv2
import numpy as np
import os
import sys
import argparse
import torch as th

from model import MattingNetwork

def segment(in_vid_path):
    if not os.path.exists(in_vid_path):
        print("error: %s not exists" % in_vid_path)
        return

    model=MattingNetwork('resnet50').eval().cuda()
    model.load_state_dict(th.load('rvm_resnet50.pth'))

    in_vid = cv2.VideoCapture(in_vid_path)
    in_vid_height = int(in_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_vid_width = int(in_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_vid_frm_num = int(in_vid.get(cv2.CAP_PROP_FRAME_COUNT))

    out_vid_path = in_vid_path[:-4] + "_out.mp4"
    out_vid_path1 = in_vid_path[:-4] + "_out1.mp4"
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(
        out_vid_path,
        fourcc,
        float(in_vid.get(cv2.CAP_PROP_FPS)),
        (in_vid_width*3, in_vid_height)
        # (in_vid_height*3, in_vid_width)
    )
    out_vid1 = cv2.VideoWriter(
        out_vid_path1,
        fourcc,
        float(in_vid.get(cv2.CAP_PROP_FPS)),
        (in_vid_width, in_vid_height)
        # (in_vid_height*3, in_vid_width)
    )

    bgr = th.tensor([.47, 1, .6]).view(3,1,1).cuda()
    rec=[None]*4
    downsample_ratio = 0.25

    with th.no_grad():
        for frm_idx in range(1200):
            success, img = in_vid.read()
            if not success:
                print("error: cannot read image")
                break
            if frm_idx%10==0:
                
                
                img = cv2.transpose(img)   
                img = cv2.flip(img, 1)
                src = np.copy(img).astype(np.float32)/255.0
                src = th.tensor(src, dtype=th.float32).cuda()
                src = src[None, ...].permute(0,3,1,2)

                fgr, pha, *rec = model(src.cuda(), *rec, downsample_ratio)

                pha = pha.permute(0,2,3,1)[0].cpu().numpy()
                # print(pha)
                # print(img.shape)
                img_b = img*pha.clip(0,1)
                # print(img_b)
                pha = pha*255.0
                pha = pha.clip(0.0, 255.0).astype(np.uint8)

                seg_count_thres = 127
                _, thres = cv2.threshold(pha, seg_count_thres, 255, 0)
                contours, _ = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                num = len(contours)
                cur_idx = -1
                cur_len = 0
                for idx in range(num):
                    seg_len = len(contours[idx])
                    if seg_len > cur_len:
                        cur_len = seg_len
                        cur_idx = idx
                if cur_idx<0:
                    return None
                else:
                    contour = contours[cur_idx].squeeze(axis=1)

                img_cont = np.copy(img)
                for pt_id in range(contour.shape[0]):
                    pt = contour[pt_id]
                    cv2.circle(img_cont, (int(pt[0]),int(pt[1])), 1, (255, 0, 0), -1)

                res_frm = np.zeros((in_vid_height, in_vid_width*3, 3), dtype=np.uint8)
                res_frm1 = np.zeros((in_vid_height, in_vid_width, 3), dtype=np.uint8)
                # res_frm = np.zeros((in_vid_height, in_vid_width*3, 3), dtype=np.uint8)
                res_frm[:, 0:in_vid_width] = img
                res_frm1[:, 0:in_vid_width] = img_b
                for i in range(3):
                    res_frm[:, in_vid_width:in_vid_width*2, i:i+1] = pha
                res_frm[:, in_vid_width*2:in_vid_width*3] = img_cont

                out_vid.write(res_frm)
                out_vid1.write(res_frm1)
                # if frm_idx == in_vid_frm_num-20:
                #     break
                print("Segment frame [%d/%d]" % (frm_idx + 1, in_vid_frm_num), end="\r")
        print("")

if __name__ == "__main__":
    segment('../../proj/output_2.mov')