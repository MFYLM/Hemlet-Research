import numpy as np
import matplotlib.pyplot as plt
import os


INDICES = [0, 24, 25, 26, 29, 30, 31, 2, 5, 6, 7, 17, 18, 19, 9, 10, 11]
edges = [[0, 1], [0, 4], [0, 7], [1, 2], [2, 3], [4, 5], [5, 6], [7, 8], [8, 9], [9, 10], [7, 11], [11, 12], [12, 13]]
lefts = [4, 5, 6, 8, 9, 10]
rights = [1, 2, 3, 11, 12, 13]

def draw_skeleton(pose, ax, is_3d=False, label = False,joints = 14):

    col_right = 'b'
    col_left = 'r'
    # print("current pose shape is: ",np.shape(pose))
    # print(pose)
    if is_3d:
        ax.scatter(pose[:, 0], pose[:, 1], zs=pose[:, 2], color='k')
    else:
        # ax.scatter(pose[:][0], pose[:][1], color='k')
        ax.scatter(pose[:, 0], pose[:, 1], color='k')

    if label:
        for i in range(joints):
            ax.annotate(i,(pose[i, 0], pose[i, 1]))
    if not label:
        for u, v in edges:
            col_to_use = 'k'

            if u in lefts and v in lefts:
                col_to_use = col_left
            elif u in rights and v in rights:
                col_to_use = col_right

            if is_3d:
                ax.plot([pose[u, 0], pose[v, 0]], [pose[u, 1], pose[v, 1]], zs=[pose[u, 2], pose[v, 2]], color=col_to_use)
            else:
                ax.plot([pose[u, 0], pose[v, 0]], [pose[u, 1], pose[v, 1]], color=col_to_use)
                # ax.plot([pose[u][0], pose[v][0]], [pose[u][1], pose[v][1]], color=col_to_use)



def visualize(joints, ipath, img_path, seq, index, width=1920, height=1080):

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        imgpath = ipath +"/"+ seq +"/"+f"image{index:05d}.jpg"
        ax.imshow(plt.imread(img_path))

        for j in range(len(joints)):

            # draw_skeleton(joints[j][i, :, :], ax,label=False)
            draw_skeleton(joints[j][index], ax,label=False)

        ax.set_xlim((0, width))
        ax.set_ylim((height, 0))

        plt.draw()
        plt.pause(0.0001)


def load_sequence(spath, seq):

        seq_file = os.path.join(spath, f"{seq}.pkl")

        # load data
        # seq_file = r"C:\Users\LangZheZR\Desktop\Research\sequenceFiles\sequenceFiles\test\downtown_bus_00.pkl"
        # print(os.path.splitext(os.listdir("C:/Users/LangZheZR/Desktop/Research/sequenceFiles/sequenceFiles/train/")[0])[0])
        data = np.load(seq_file, encoding='latin1', allow_pickle=True)
        intrinsic = data['cam_intrinsics']
        valid_data = data["campose_valid"]
        subjects = data['jointPositions']

        f = np.array([intrinsic[0, 0], intrinsic[1, 1]]).reshape((2, 1))
        c = np.array(intrinsic[:2, 2]).reshape((2, 1))

        jnts_2d = []

        # what is the subject mean in this?
        # subject mean the position of people here
        for i_sub in range(len(subjects)):

            subject = subjects[i_sub]
            jnts_2d.append([])

            for i_frame in range(subject.shape[0]):
                # skip invalid frame
                # if not valid_data[i_sub][i_frame]:
                #     # jnts_2d[i_sub].append([0,0])
                #     print(valid_data[i_sub][i_frame])
                #     continue

                cam_pose = data['cam_poses'][i_frame, :, :]
                R = cam_pose[:3, :3]
                T = cam_pose[:3, 3].reshape((3, -1))

                pos3d_world = subject[i_frame].reshape((24, 3)) [INDICES, :]
                pos3d_world_h = np.vstack((pos3d_world.T, np.ones((1, 14)))) #origin (1,14)

                pos3d_cam = np.matmul(cam_pose, pos3d_world_h).T[:, :3]
                proj_2d = np.divide(pos3d_cam[:, :2], pos3d_cam[:, 2:])
                pixel_2d = f[:, 0] * proj_2d + c[:, 0]

                jnts_2d[i_sub].append(pixel_2d)

            jnts_2d[i_sub] = np.array(jnts_2d[i_sub])#,dtype= np.float)

        assert len(jnts_2d) == len(subjects)
        joints = np.array(jnts_2d)#,dtype= np.float)
        return joints, c*2


