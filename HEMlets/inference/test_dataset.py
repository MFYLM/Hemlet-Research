import cv2
import numpy as np
import os
import sys

import torch
import torch.utils.data as tData
import glob

import random
import json 
import sys 
sys.path.append('..')
import h5py  
import poseutils.props

import scipy.io
import torch
import cv2
from PIL import Image
import pylab
import torchvision.transforms as transforms
import torchvision.transforms as T
import numpy
import scipy.io as sio
from table import *
import matplotlib
import matplotlib.pyplot as plt

class H36M(tData.Dataset):
    def __init__(self,path,video_id=1,subject=11,patch_width=256,patch_height=256,split = 'train'):

        super(H36M,self).__init__()
        self.index = [0,2,5,8,1,4,7,9,12,6,15,13,18,20,14,19,21,0]

        self.subject = subject

        self.path = path
        self.video_id = video_id
        self.image_conter = 0
        self.image_lst = []
        self.info_lst = []
        for folder in os.listdir(self.path):
            f = os.path.join(self.path, folder)
            for filename in os.scandir(path = f):
                file_path = os.path.join(f, filename)
                
                if "mp4" in str(filename):
                    core_name = str(filename).split(".")
                    core_name = core_name[0]
                    core_name = core_name.split("'")
                    core_name = core_name[1]
                    info_file_name = core_name+"_info.mat"
                    info_file_path = os.path.join(f, info_file_name)
                    if os.path.exists(info_file_path):
                        self.image_lst.append(file_path)
                        self.info_lst.append(info_file_path)

        self.len_data = len(self.image_lst)

        print(self.len_data)

    def __len__(self):
        return self.len_data 
    
    def imgNormalize(self,img,flag = True):
        if flag:
            img = img[:,:,[2,1,0]]
        return np.divide(img - img_mean, img_std)
        
        
        
    def __getitem__(self, index):
        # index = 555
        print("##index: ", index)
        idx = index % self.__len__()
        #load image tensor, info mat
        vidcap = cv2.VideoCapture(self.image_lst[index])
        success,image = vidcap.read()
        info = sio.loadmat(self.info_lst[index])
        
        pylab.imshow(image)
        pylab.show()
        joints2D = info["joints2D"]
        joints2D = np.moveaxis(joints2D, -1, 0)
        joints2D = np.moveaxis(joints2D, -1, 1)
        joints2D =  np.array([joints2D[0]])

        box = poseutils.props.get_bounding_box_2d(joints2D)
        lx,ly,rx,ry = int(box[0]),int(box[1]),int(box[2]),int(box[3])

        crop = image[ly:ry,lx:rx]
        # padding
        padding = abs((rx-lx)-(ry-ly))
        if (rx-lx) > (ry-ly):
            padd_img = cv2.copyMakeBorder(crop,padding,0,0,0,cv2.BORDER_CONSTANT)
        else:
            padd_img = cv2.copyMakeBorder(crop,0,0,0,padding,cv2.BORDER_CONSTANT)

        #resize to 256x256
        dim = (256,256)
        try:
            assert padd_img is not None
            img = cv2.resize(padd_img, dim, interpolation = cv2.INTER_AREA)
        except: 
            print("invalid resize image file is: ",img_path)
            img = np.ndarray(shape=(256,256),dtype = float)
#         pylab.imshow(img)
#         pylab.show()
        #converted image to same size
#         img = np.moveaxis(img, [0, 1, 2], [1, 2, 0])

        image = self.imgNormalize(img)
        image = np.transpose(image,(2,0,1))

        image_filp = image[:,:,::-1].copy()
        image_filp = torch.from_numpy(image_filp).float()

        image = torch.from_numpy(image).float()    
        
        #load info tensor
        joints = info["joints3D"][:,0,self.index].reshape(18,3)
        joints[:,2] = joints[:,2] / 255.0 - 0.5
        joints[:,0:2] = joints[:,0:2] / 256.0 - 0.5
        joint3d = torch.from_numpy(joints).float()
#         print(joint3d.shape)
        return image,image_filp,joint3d


#         joint3d_j18 = info["joints3D"][:,0,self.index].reshape(18,3)
#         joint3d_j18[17] = (joint3d_j18[11] + joint3d_j18[14]) * 0.5 
#         joint3d_j18 = torch.from_numpy(joint3d_j18).float()

#         return image,image_filp,joint3d_j18
        
if __name__ == '__main__':
    matplotlib.use('GTKAgg')
    d = H36M(split = 'val')
    for _ in range(100):
        d[12000]
        input('check')

