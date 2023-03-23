from dpw_loader import dpw
import dataloader 
from config import config
from network import Network
from model_opr import load_model
import torch
import cv2
import os
import sys
import numpy as np
sys.path.append('./')
from table import *
from draw_figure import *
from getActionID import *
from metric_3d import * 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.gridspec as gridspec
import imageio_ffmpeg

import argparse

"""
RuntimeError: Given groups=1, weight of size [64, 3, 7, 7], expected input[1, 256, 256, 3] to have 3 channels, but got 256 channels instead  
"""

joint14_HEMlet_index = [0,1,2,3,4,5,6,8,11,12,13,14,15,16]

model = Network(config)
# device = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

seq_path =  "C:/Users/LangZheZR/Desktop/Research/sequenceFiles/sequenceFiles/test"
img_path = "C:/Users/LangZheZR/Desktop/Research/imageFiles"
checkpoint_path = r'./ckpt/hemlets_h36m_lastest.pth'

load_model(model,checkpoint_path,cpu=not torch.cuda.is_available())
model.eval()
val_dataset = dpw(seq_path=seq_path,img_path=img_path, single= True)
val_loader = torch.utils.data.DataLoader(val_dataset)
# val_loader = dataloader.val_loader(val_dataset,config,0,1)
print('finish running')

# N_viz = val_loader.__len__()
# for idx, data in enumerate(val_loader):
#     print("running.....")
#     if idx>=N_viz:
#         break
#     image = data
#     image = image.to(device)
#     with torch.no_grad():
#         pred_joint3d = model(image,val= True)
#     print(pred_joint3d)
def MPJPE_P1_P2_dpw(pose_a,pose_b):
    pose_a_j14 = pose_a
    pose_b_j14 = pose_b
    
    pose_a_j14 = pose_a_j14.reshape((-1,3))
    pose_b_j14 = pose_b_j14.reshape((-1,3))

    pose_a_j14 = move_hip_to_origin_dpw(pose_a_j14)
    pose_b_j14 = move_hip_to_origin_dpw(pose_b_j14)

    error = np.average(np.sqrt(np.sum(np.square(pose_a_j14 - pose_b_j14),axis=-1))) 
    return error

def move_hip_to_origin_dpw(pose):
    rootPos = pose[0]
    poseNew = np.zeros((pose.shape[0],3),dtype=float)
    for i in range(pose.shape[0]):
        poseNew[i,:] = pose[i,:] - rootPos
    return  poseNew

# def normalized_to_original(image):
#     image_numpy = image.cpu().numpy()
#     image_numpy = np.transpose(image_numpy, (0, 2, 3, 1))
#     image_numpy = image_numpy * img_std + img_mean
#     return image_numpy.astype(np.uint8)

def from_normjoint_to_cropspace(joint3d):
    joint3d[:,:,:2] = (joint3d[:,:,:2] + 0.5 )*256.0
    return joint3d

def dpw_eval_matric(joint_pred,joint_flip):
    joint_pred_crop = from_normjoint_to_cropspace(joint_pred)
    joint_flip_crop = from_normjoint_to_cropspace(joint_flip)
    
    patch_width = 256.0
    crop_pred_j3d = joint_pred_crop[0]
    pipws_flip = joint_flip_crop[0]
    pipws_flip[ :, 0] = patch_width - pipws_flip[ :, 0] - 1
    for pair in flip_pairs:
            tmp = pipws_flip[ pair[0], :].copy()
            pipws_flip[ pair[0], :] = pipws_flip[ pair[1], :].copy()
            pipws_flip[ pair[1], :] = tmp.copy()
    mixJoint = (pipws_flip + crop_pred_j3d) * 0.5
    return mixJoint

def DrawGTSkeleton(channels,ax):
    edges = [[0, 1], [0, 4], [0, 7], [1, 2], [2, 3], [4, 5], [5, 6], [7, 8], [8, 9], [9, 10], [7, 11], [11, 12], [12, 13]]
    edge = np.array(edges)
    I    = edge[:,0]
    J    = edge[:,1]
    LR  = np.ones((edge.shape[0]),dtype=np.int)
    colors = [(0,0,1.0),(0,1.0,0),(1.0,0,0)]
    vals = np.reshape( channels, (-1, 3) )

    vals[:] = vals[:] - vals[0]

    ax.cla()
    ax.view_init(azim=-136,elev=-157)
    ax.invert_yaxis()

    for i in np.arange( len(I) ):
        x,y,z = [np.array([vals[I[i],j],vals[J[i],j]]) for j in range(3)]
        ax.plot(x, -z, y, lw=2)

    # for i in range(14):
    #     ax.plot([vals[i,0],vals[i,0]+1],[-vals[i,2],-vals[i,2]],[vals[i,1],vals[i,1]],lw=3,c=(0.0,0.8,0.0))	

def cropPoseToFullPose(cropPose,trans):
    fullPose = cropPose.copy()
    fullPose = fullPose / trans[2]
    fullPose[:,0] = fullPose[:,0] + trans[0]
    fullPose[:,1] = fullPose[:,1] + trans[1]
    return fullPose
    
def dpw_invPoseToCamSpacePlus(mixjoint, trans_np):
    fullPose2d = cropPoseToFullPose(mixjoint[:,0:2],trans_np)
    j3d = restore_cameraspace_3d_joints(fullPose2d,mixjoint[:,2]  * 2000.0)
    return j3d,j3d,fullPose2d
    
def dpw_validate(model, val_loader, device):
    
    #visualize set up : 
    font = {'family' : 'serif',  
            'color'  : 'darkred',  
            'weight' : 'normal',  
            'size'   : 10,  
                }
    fig = plt.figure( figsize=(19.2 / 2, 10.8 / 2) )
    gs1 = gridspec.GridSpec(2, 3)
    gs1.update(left=0.08, right=0.98,top=0.95,bottom=0.08,wspace=0.05, hspace=0.1)
    axImg=plt.subplot(gs1[0,0])
    ax_true_pose = plt.subplot(gs1[0,1],projection='3d')
    axImg.axis('off')
    axPose3d_pred=plt.subplot(gs1[0,2],projection='3d')
    
    N_viz = val_loader.__len__()
    for idx, data in enumerate(val_loader):
        if idx>=N_viz:
            break
        image, image_flip, original_img, true_joint, trans = data
        image = image.to(device)
        with torch.no_grad():
            pred_joint3d = model(image,val= True)
            pred_joint3d_flip = model(image_flip,val=True)
        
        
        pre_3d_np = pred_joint3d.cpu().numpy()
        pre_3d_flip_np = pred_joint3d_flip.cpu().numpy()
        pred_crop_3d_joint = dpw_eval_matric(pre_3d_np,pre_3d_flip_np)
        Draw3DSkeleton(pred_crop_3d_joint,axPose3d_pred,JOINT_CONNECTIONS,'Pred_joint3d',fontdict=font,j18_color=JOINT_COLOR_INDEX,image = None)
        axImg.imshow(original_img[0])
        axImg.axis('off')
        
        # show the scatter of true_joint_position 
        print("Shape of number of joint: ",true_joint[0]," shape of true: ", np.shape(true_joint[0]))
        DrawGTSkeleton(true_joint[0],ax_true_pose)
        
        pred_cam3d_unity,_,_ = dpw_invPoseToCamSpacePlus(pred_crop_3d_joint,trans[0].cpu().numpy())
        error = MPJPE_P1_P2_dpw(pred_cam3d_unity[joint14_HEMlet_index,:],true_joint[0])
        print("########################################")
        print("####the euclidean distance: " ,error)
        print("########################################")

        
        plt.draw()
    
        plt.pause(0.0001)
        
        
dpw_validate(model,val_loader,device)


"""
Calculate for Euclidean distance    
"""


