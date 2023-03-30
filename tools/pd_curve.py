import json
import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import re
import cv2

def CameraPoseRead(camera_name):

    camera_pose_path = camera_name
    camera_pose = []

    f = open(camera_pose_path)
    for i in range(4):
        line = f.readline()
        tmp = line.split()
        camera_pose.append(tmp)
    camera_pose = np.array(camera_pose, dtype=np.float32)

    return camera_pose

if __name__ == '__main__':
    depthDir = '/media/qjx/Elements/scene0000_00/depth'
    depth_paths = [os.path.join(depthDir,dir) for dir in os.listdir(depthDir) if '.png' in dir] 
    depth_paths.sort()
    poseDir = '/media/qjx/Elements/scene0000_00/pose'
    pose_paths = [os.path.join(poseDir,dir) for dir in os.listdir(poseDir) if '.txt' in dir] 
    pose_paths.sort()
    posezs = []
    depthms = []
    z_min = 0xffff
    z_max = -0x0fff 
    for i in range(len(pose_paths)):
        print(i)
        # if i<1400:
        #     continue
        # if i>1500:
        #     break
        pose = CameraPoseRead(pose_paths[i])
        pose = np.linalg.inv(pose)
        posez = -pose[2,3]
        if posez < z_min:
            z_min = posez
        elif posez > z_max:
            z_max = posez
        posezs.append(posez)
        depth = cv2.imread(depth_paths[i], -1) * 1.0 / 1000
        depth = np.where(depth>5.0,0.0,depth)
        depthm = np.max(depth)
        depthms.append(depthm)
    
    idx = range(0,len(posezs))
    
    print(z_min,z_max)
    # exit(0)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(idx, depthms, 'r', label="depth");
    ax1.legend(loc=1)
    ax1.set_ylabel('Depth');
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(idx, posezs, 'g', label="pose_z")
    ax2.legend(loc=2)
    ax2.set_ylabel('pose_z')
    ax2.set_xlabel('idx')
    plt.show()

    

