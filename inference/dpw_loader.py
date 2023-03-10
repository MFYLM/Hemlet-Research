import numpy as np
import os
import sys
import torch
import torch.utils.data as tData
import sys 
import poseutils.props
import matplotlib.pyplot as plt
import cv2 as cv
# from config import config
sys.path.append('..')

"""_summary_
1. are we allowed to use the same img_mean and std?

2. i got this warning when I am trying to skip the invalid frame
    VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
    return np.array(jnts_2d), invalidFrame

"""


class dpw(tData.Dataset):
    def __init__(self,seq_path,img_path,single = False):

        super(dpw,self).__init__()
        self.single = single
        self.index = [0, 2, 5, 8, 1, 4, 7, 12, 16, 18, 20, 17, 19, 21]
        self.img_mean = np.array([123.675,116.280,103.530])
        self.img_std = np.array([58.395,57.120,57.375])
        self.img_path = img_path
        self.seq_path = seq_path
        self.all_imgs = []
        self.singleJoints = [] # only works for single subject
        self.ground_truth_coordinate = []
        self.all_boundaries = []
        self.loadAllImageAndBoxes()
        
    def __len__(self):
        return len(self.all_imgs)

    def imgNormalize(self,img,flag = True):
        if flag:
            img = img[:,:,[2,1,0]]
        return np.divide(img - self.img_mean, self.img_std)

    def loadAllImageAndBoxes(self):
        seq_names = os.listdir(self.seq_path)
        for cur_name in seq_names:
            cur_seq_file = os.path.join(self.seq_path,cur_name)
            cur_img_folder = os.path.join(self.img_path,os.path.splitext(cur_name)[0])
            images = os.listdir(cur_img_folder)
            joints_2d, invalid_f = self.loadJoints(cur_seq_file)
            
            cur_boxes = []
            if not self.single:
                for sub_i in range((len(joints_2d))):
                    box = poseutils.props.get_bounding_box_2d(joints_2d[sub_i])
                    cur_boxes.append(box)
            else: # single subject
                cur_boxes.append(poseutils.props.get_bounding_box_2d(joints_2d[0]))
                
            # the joints and boxes only contains valid frames
            # if invalid frames exist, image still contain invalid frames
            # need invalid_f to make sure images and boundaries have same length
            valid_index = 0
            for p in range(len(images)):
                if p not in invalid_f:
                    self.all_boundaries.append((self.findCombineBox(valid_index,cur_boxes)))
                    self.all_imgs.append(os.path.join(cur_img_folder,images[p]))
                    valid_index+=1
            
    def loadJoints(self,seq_file):
        data = np.load(seq_file, encoding='latin1', allow_pickle=True)
        intrinsic = data['cam_intrinsics']
        valid_data = data["campose_valid"]
        subjects = data['jointPositions']
        f = np.array([intrinsic[0, 0], intrinsic[1, 1]]).reshape((2, 1))
        c = np.array(intrinsic[:2, 2]).reshape((2, 1))
        jnts_2d = []

        # all subject append same frames
        checker = []
        
        #get the real world 
        # this does not looks right
        # self.singleJoints.append(subjects[0])

        # subject mean the position of people here
        invalidFrame = set()
        for sb in range(len(subjects)):
            subject = subjects[sb]
            for fra in range(subject.shape[0]):
                if not valid_data[sb][fra]:
                    invalidFrame.add(fra)
        
        assert len(set(len(s) for s in valid_data)) == 1, f"each subjects frame number are different! {len(val_data[0])}"
        for i_sub in range(len(subjects)):
            subject = subjects[i_sub]
            jnts_2d.append([])
            self.singleJoints.append([])
            cur_sub = 0
            for i_frame in range(subject.shape[0]):
                # skip invalid frame
                # there are invalid frame in second but the first don't know
                if i_frame in invalidFrame:
                #     invalidFrame.add(i_frame)
                    continue

                cam_pose = data['cam_poses'][i_frame, :, :]
                R = cam_pose[:3, :3]
                T = cam_pose[:3, 3].reshape((3, -1))
                pos3d_world = subject[i_frame].reshape((24, 3))[self.index, :]
                pos3d_world_h = np.vstack((pos3d_world.T, np.ones((1, 14))))
                pos3d_cam = np.matmul(cam_pose, pos3d_world_h).T[:, :3]
                proj_2d = np.divide(pos3d_cam[:, :2], pos3d_cam[:, 2:])
                pixel_2d = f[:, 0] * proj_2d + c[:, 0]
                jnts_2d[i_sub].append(pixel_2d)
                self.singleJoints[i_sub].append(pos3d_cam)
                cur_sub +=1
            checker.append(cur_sub)
            # self.singleJoints[i_sub] = np.array(self.singleJoints[i_sub])
            jnts_2d[i_sub] = np.array(jnts_2d[i_sub])
        
        # to make sure all subjects has same length 
        assert len(set(s for s in checker)) == 1, f"number for each subject's frame are different! : {checker}"
        
        assert len(jnts_2d) == len(subjects)
        return np.array(jnts_2d), invalidFrame
    
    def findCombineBox(self,p_i,boxes):
        boundingBox = {"left_x": float('inf'),          #0
                       "left_y": float('inf'),          #1
                       "right_x": -float('inf'),        #2
                       "right_y": -float('inf')}        #3
        for s in range(len(boxes)): # for each human in that picture
            boundingBox["left_x"] = min(boundingBox["left_x"],boxes[s][0][p_i])
            boundingBox["left_y"] = min(boundingBox["left_y"],boxes[s][1][p_i])
            boundingBox["right_x"] = max(boundingBox["right_x"],boxes[s][2][p_i])
            boundingBox["right_y"] = max(boundingBox["right_y"],boxes[s][3][p_i])
        return int(boundingBox["left_x"]),int(boundingBox["left_y"]),int(boundingBox["right_x"]),int(boundingBox["right_y"])
    
    def resizeImg(self,img_path,ind): # return 256X256 img
        img = plt.imread(img_path)
        lx,ly,rx,ry = self.all_boundaries[ind]
        # crop 
        crop = img[ly:ry,lx:rx]
        
        # padding
        padding = abs((rx-lx)-(ry-ly))
        if (rx-lx) > (ry-ly):
            padd_img = cv.copyMakeBorder(crop,padding,0,0,0,cv.BORDER_CONSTANT)
        else:
            padd_img = cv.copyMakeBorder(crop,0,0,0,padding,cv.BORDER_CONSTANT)
        
        #resize to 256x256
        dim = (256,256)
        try:
            assert padd_img is not None
            return cv.resize(padd_img, dim, interpolation = cv.INTER_AREA)
        except: 
            print("invalid resize image file is: ",img_path)
            return np.ndarray(shape=(256,256,3),dtype = float)
        
    def __getitem__(self, index):
        img = self.resizeImg(self.all_imgs[index],index)
        image = self.imgNormalize(img)
        image = np.transpose(image,(2,0,1))
        image_filp = image[:,:,::-1].copy()
        image_filp = torch.from_numpy(image_filp).float()
        image = torch.from_numpy(image).float()
        return image, image_filp, img, np.array(self.singleJoints[0][index])

if __name__ == '__main__':
    seq_path =  "C:/Users/LangZheZR/Desktop/Research/sequenceFiles/sequenceFiles/test"
    img_path = "C:/Users/LangZheZR/Desktop/Research/imageFiles"
    val_data = dpw(seq_path=seq_path,img_path=img_path)
    seq_file = r"C:/Users/LangZheZR/Desktop/Research/sequenceFiles/sequenceFiles/downtown_bus_00"
    joints, useless = val_data.loadJoints(seq_file)
    

    
    file =r"C:/Users/LangZheZR/Desktop/Research/imageFiles\downtown_bus_00\image_01645.jpg"
    img = plt.imread(file,0)
    dim = (256,256)
    resize = cv.resize(img, dim, interpolation = cv.INTER_AREA)

    f, sub = plt.subplots(figsize = (15,15))
    # sub[0].imshow(img)
    # sub[0].set_title("paddimg")
    
    sub.imshow(resize)
    sub.set_title("resize")
    print(1400//700)