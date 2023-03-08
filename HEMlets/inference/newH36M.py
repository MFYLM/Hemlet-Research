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

from table import *

class H36M(tData.Dataset):
    def __init__(self,path,video_id=1,subject=11,patch_width=256,patch_height=256,split = 'train'):

        super(H36M,self).__init__()

        self.subject = subject

        self.path = path

        self.video_id = video_id
        
        self.image_lst = []
        for filename in os.scandir(path = f):
            file_path = os.path.join(f, filename)
            if os.path.isfile(file_path):
                #read image
                vidcap = cv2.VideoCapture(path)
                success,image = vidcap.read()
                count = 0
                # print("here")
                cv2.imwrite("image%d.jpg" % count, image)
                image = Image.open("image%d.jpg" % count)
                # image = Image.open("here%d.jpg" % count)
                # Define a transform to convert PIL 
                # image to a Torch tensor
                transform = transforms.Compose([
                    transforms.PILToTensor()
                ])
                transform_to_tensor = transforms.PILToTensor()
                transform = T.Resize(size = (256,256))
                # apply the transform on the input image
                image = transform(image)
                img_tensor = transform_to_tensor(image)
                # print(img_tensor)
                img_tensor = torch.movedim(img_tensor, (1, 2), (0, 1))
                image_lst.append(img_tensor)

        with h5py.File(self.path,'r') as db:
            self.len_data = db['images'].shape[0]

        print(self.len_data)

    def openfile(self):
        for filename in os.listdir(self.path):
        f = os.path.join(self.path, filename)

        for filename in os.scandir(path = f):
            file_path = os.path.join(f, filename)
            if os.path.isfile(file_path):
                return file_path
       
        return 'no valid file'
    
    def __len__(self):
        return self.len_data 

    def imgNormalize(self,img,flag = True):
        if flag:
            img = img[:,:,[2,1,0]]
        return np.divide(img - img_mean, img_std)

    def __getitem__(self, index):
        # index = 555
        h5_path =  self.h5_path
        print("##index: ", index)
        idx = index % self.__len__()
        with h5py.File(h5_path,'r') as db:

            joints3dCam = db['joints3d_cam'][idx] # load the camera space 3d joint (32,3)
            joint3d_j18 = np.zeros((18,3),dtype=float)
            joint3d_j18[0:17,:] = joints3dCam[H36M_TO_J18,:] 
            joint3d_j18[17] = (joint3d_j18[11] + joint3d_j18[14]) * 0.5 


            img = self.image_lst[idx]
            joints = db['joints'][idx]
            
            trans = db['trans'][idx]
            camid = db['camid'][idx]

            cam_id = np.zeros((4,),dtype = int)
            cam_id[:3] = camid
            cam_id[3] = self.video_id #int( ((h5_path.split('/')[-1]).split('.')[0]).split('_')[4])


            joints[:,2] = joints[:,2] / 255.0 - 0.5
            joints[:,0:2] = joints[:,0:2] / 256.0 - 0.5

            image = self.imgNormalize(img)

            joint3d = torch.from_numpy(joints).float()
            joint3d_j18 = torch.from_numpy(joint3d_j18).float()
            image = np.transpose(image,(2,0,1))

            image_filp = image[:,:,::-1].copy()
            image_filp = torch.from_numpy(image_filp).float()
           
            image = torch.from_numpy(image).float()

            
            return image,image_filp,trans,cam_id,joint3d,joint3d_j18,np.zeros((3),dtype=float),'subject_{}'.format(self.subject)
if __name__ == '__main__':
    d = H36M(split = 'val')
    for _ in range(100):
        d[12000]
        input('check')
