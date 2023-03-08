import cv2
import os
import torch
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

import time 
def normalized_to_original(image):
    image_numpy = image.cpu().numpy()
    image_numpy = np.transpose(image_numpy, (0, 2, 3, 1))
    image_numpy = image_numpy * img_std + img_mean
    return image_numpy.astype(np.uint8)
def plot_2d_skeleton_in_3d(labels):
    """
    labels: 3xN
    """
    labels = np.moveaxis(labels, -1, 0)
    x = labels[0]
    y = labels[1]
    z = labels[2]
    ax.set_aspect('auto')
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, cmap='Greens');
    
    connections = [
        [0,1],
        [1,2],
        [2,3],
        [0,4],
        [4,5],
        [5, 6],
        [0,8],
        [8,9],
        [9,10],
        [7,11],
        [11,12],
        [12,13],
        [7,14],
        [14,15],
        [15,16],
    ]
    colors = ['k','k','k','r','r','r','b','b','b','r','r','r', 'k','k','k']
        
    for i, (start, end) in enumerate(connections):
        ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], c=colors[i])
    
    plt.show()
    
def validate(model, val_loader, device, subject=9,visualize = False):
    gs1 = gridspec.GridSpec(2, 3) # 6 rows, 10 columns
    gs1.update(left=0.08, right=0.98,top=0.95,bottom=0.08,wspace=0.05, hspace=0.1)
    axPose3d_gt=plt.subplot(gs1[0,1],projection='3d')

    N_viz = val_loader.__len__() 
    idx_list = []
    for idx, data in enumerate(val_loader):
        if idx >= N_viz:
            break
        #image,image_flip, trans, camid, joint3d, joint3d_camera,  root, name = data
        image,image_flip, joint3d = data
        image = image.to(device)
        # for flip test to improve the accuracy of 3d pose prediction 
        image_flip = image_flip.to(device)
        joint3d = joint3d.to(device)
        # start_time = time.time()

        with torch.no_grad():
#             print(image.shape)
#             print(image_flip.shape)
            pred_joint3d = model(image,val=True)
            pred_joint3d_flip = model(image_flip,val=True)

        pred_joint3d_flip_numpy = pred_joint3d_flip.cpu().numpy()
        pred_joint3d_numpy = pred_joint3d.cpu().numpy()

#         print(pred_joint3d_numpy.shape)
#         print(pred_joint3d_numpy)
#         print(joint3d.shape)
#         print(joint3d)
        # normalized (0-1) 3d joints
        gt_joint3d_numpy = joint3d.cpu().numpy()
          

#         # calculate the MPJPE(Protocol #1) and MPJPE(Protocol #2)
        gt_crop_3d_joint,pred_crop_3d_joint= \
                    eval_metric(pred_joint3d_numpy,pred_joint3d_flip_numpy,gt_joint3d_numpy,\
                    debug = False,return_viz_joints=True)
        plot_2d_skeleton_in_3d(pred_crop_3d_joint)
        
#         Draw3DSkeleton(gt_crop_3d_joint,axPose3d_gt,JOINT_CONNECTIONS,'GT_joint3d',\
#                        fontdict=font,j18_color=JOINT_COLOR_INDEX,image = None)
#         Draw3DSkeleton(pred_crop_3d_joint,axPose3d_pred,JOINT_CONNECTIONS,'Pred_joint3d',\
#                        fontdict=font,j18_color=JOINT_COLOR_INDEX,image = None)
        
        
#         print("here",gt_crop_3d_joint.shape)
#         plot_2d_skeleton_in_3d(pred_crop_3d_joint)
#         print("here",pred_crop_3d_joint)
    
    
#         if visualize:
#             # convert image tensor to image numpy
#             image_batch = normalized_to_original(image)
#             img = image_batch[0]
#             # draw gt 2d joint on the image  
#             image_draw = drawSkeleton(img.copy(),gt_crop_3d_joint,JOINT_CONNECTIONS,JOINT_COLOR_INDEX)
#             plt.show()
#             # draw 3d joints


#             plt.draw()             
#     if visualize:
#         vw_pure.close()


#     for k,_ in action_wise_error_dict.items():
#         if action_wise_error_dict[k][2]>0:
#             print(actions[k],action_wise_error_dict[k][0]/action_wise_error_dict[k][2],action_wise_error_dict[k][1]/action_wise_error_dict[k][2])


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="HEMlets(pose) inference script")
    parser.add_argument("--ckpt_path", type=str,\
                        default="./ckpt/hemlets_h36m_lastest.pth", \
                        help='path to model of the pretrained model')
    parser.add_argument("--dataset_path", type=str,\
                        default="./data/S11/S_11_C_4_1_full.h5", \
                        help='path to a dataset')
    parser.add_argument("--video_id", type=int,\
                        default=1, \
                        help='video id (1-120) pre sequence of Human3.6M')
    parser.add_argument("--visualize", type=int,\
                        default=0, \
                        help='activate the function of visualize')
    parser.add_argument("--sequence_id", type=int,\
                        default=11, \
                        help='evaluation sequence id of the testing dataset')

    argspar = parser.parse_args()
    print('argspar',argspar)

    from config import config
    from network import Network
    import dataloader
    from model_opr import load_model
    from test_dataset import H36M

    # define network 
    model = Network(config)
    # device = torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)


    checkpoint_path = argspar.ckpt_path # './ckpt/hemlets_lastest.pth'
    tiny_dataset = argspar.dataset_path  #'./data/S11/S_11_C_4_1_full.h5'
    print('tiny_dataset',tiny_dataset)
    video_id = argspar.video_id # (1-120)

    visualize = True if argspar.visualize==1 else False
    print('argspar.visualize',argspar.visualize)


    # load model weights
    load_model(model, checkpoint_path, cpu=not torch.cuda.is_available())
    model.eval()


    subject = argspar.sequence_id #11


    # define dataset and dataloader
    foler_path = "/extra/wayne1/preserve/yanranw1/dataset/surreal/SURREAL/data/cmu/test/run0"
    val_dataset = H36M(path = foler_path,video_id=video_id,subject=subject, split='val')
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1)
#     for idx, data in enumerate(val_loader):
#         print(idx)
#         print(data[0].shape, data[1].shape,data[2].shape)
    # start evaluation...
    print(validate(model, val_loader, device, subject=subject,visualize=visualize))
