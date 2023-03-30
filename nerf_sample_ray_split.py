import numpy as np
from collections import OrderedDict
import torch
import cv2
import imageio
import skimage
import skimage.io
import skimage.transform
from utils import TINY_NUMBER, HUGE_NUMBER
def project_image_to_rect(uv_depth, P):
    ''' Input: nx3 first two channels are uv, 3rd channel
               is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    '''

    c_u = P[0, 2]
    c_v = P[1, 2]
    f_u = P[0, 0]
    f_v = P[1, 1]
    b_x = P[0, 3] / (-f_u)  # relative
    b_y = P[1, 3] / (-f_v)

    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
    y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
    pts_3d_rect = np.zeros((n, 3), dtype=uv_depth.dtype)
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]
    return pts_3d_rect

########################################################################################################################
# ray batch sampling
########################################################################################################################

def get_rays_single_image(H, W, intrinsics, c2w):
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    '''
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)

    rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
    rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = rays_d.transpose((1, 0))  # (H*W, 3)

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

    depth2 = np.linalg.inv(c2w)[2, 3]
    depth2 = depth2 * np.ones((rays_o.shape[0],), dtype=np.float32)

    return rays_o, rays_d, depth2

class RaySamplerSingleImage(object):
    def __init__(self, H, W, intrinsics, c2w,
                       img_path=None,
                       depth_path=None,
                       resolution_level=1, #scan:2
                       mask_path=None,
                       min_depth_path=None,
                       max_depth=None,test_flag=False):
        super().__init__()
        self.W_orig = W
        self.H_orig = H
        self.intrinsics_orig = intrinsics
        self.c2w_mat = c2w

        self.img_path = img_path
        self.depth_path = depth_path
        self.mask_path = mask_path
        self.min_depth_path = min_depth_path
        self.max_depth = max_depth

        self.resolution_level = -1
        self.test_flag = test_flag
        self.set_resolution_level(resolution_level)

    def set_resolution_level(self, resolution_level):
        if resolution_level != self.resolution_level:
            self.resolution_level = resolution_level
            self.W = self.W_orig // resolution_level
            self.H = self.H_orig // resolution_level
            self.intrinsics = np.copy(self.intrinsics_orig)
            self.intrinsics[:2, :3] /= resolution_level
            # only load image at this time
            if self.img_path is not None:
                self.img = imageio.imread(self.img_path).astype(np.float32) / 255.
                self.img = cv2.resize(self.img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                self.img = self.img.reshape((-1, 3))
            else:
                self.img = None
            
            if self.depth_path is not None:
                self.depth = cv2.imread(self.depth_path,-1).astype(np.float32) / 60000.
                self.depth = cv2.resize(self.depth, (self.W, self.H))
                self.depth = self.depth.reshape((-1,)).clip(0.0,1.0)
            else:
                self.depth = None

            if self.mask_path is not None:
                self.mask = imageio.imread(self.mask_path).astype(np.float32) / 255.
                self.mask = cv2.resize(self.mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                self.mask = self.mask.reshape((-1))
            else:
                self.mask = None

            if self.min_depth_path is not None:
                self.min_depth = imageio.imread(self.min_depth_path).astype(np.float32) / 255. * self.max_depth + 1e-4
                self.min_depth = cv2.resize(self.min_depth, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                self.min_depth = self.min_depth.reshape((-1))
            else:
                self.min_depth = None

            self.rays_o, self.rays_d, self.depth2 = get_rays_single_image(self.H, self.W,
                                                                         self.intrinsics, self.c2w_mat)
    def get_img(self):
        if self.img is not None:
            return self.img.reshape((self.H, self.W, 3))
        else:
            return None

    def get_all(self):
        if self.min_depth is not None:
            min_depth = self.min_depth
        else:
            min_depth = 1e-4 * np.ones_like(self.rays_d[..., 0])
        
        ret = OrderedDict([
            ('ray_o', self.rays_o),
            ('ray_d', self.rays_d),
            ('depth', self.depth2),
            ('rgb', self.img),
            ('mask', self.mask),
            ('min_depth', min_depth)
        ])
        # return torch tensors
        for k in ret:
            if ret[k] is not None:
                ret[k] = torch.from_numpy(ret[k])
        return ret

    def random_sample(self, N_rand, center_crop=False):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''
        if center_crop:
            quad_H = self.H - 20
            quad_W = self.W - 20

            # pixel coordinates
            u, v = np.meshgrid(np.arange(20, quad_W),
                               np.arange(20, quad_H))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = np.random.choice(u.shape[0], size=(N_rand,), replace=False)

            # Convert back to original image
            select_inds = v[select_inds] * self.W + u[select_inds]
        else:
            # Random from one image
            select_inds = np.random.choice(self.H*self.W, size=(N_rand,), replace=False)

        rays_o = self.rays_o[select_inds, :]    # [N_rand, 3]
        rays_d = self.rays_d[select_inds, :]    # [N_rand, 3]
        depth2 = self.depth2[select_inds]         # [N_rand, ]
        
        if self.img is not None:
            rgb = self.img[select_inds, :]          # [N_rand, 3]
        else:
            rgb = None

        if self.depth is not None:
            depth = self.depth[select_inds] 
            depth = depth * 9.0 + 1.0                                                                   
        else:
            depth = None

        if self.mask is not None:
            mask = self.mask[select_inds]
        else:
            mask = None

        if self.min_depth is not None:
            min_depth = self.min_depth[select_inds]
        else:
            min_depth = 1e-4 * np.ones_like(rays_d[..., 0])

        ret = OrderedDict([
            ('ray_o', rays_o),
            ('ray_d', rays_d),
            ('depth', depth2),
            ('depthr', depth),
            ('rgb', rgb),
            ('mask', mask),
            ('min_depth', min_depth),
            ('img_name', self.img_path)
        ])
        # return torch tensors
        for k in ret:
            if isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k])

        return ret