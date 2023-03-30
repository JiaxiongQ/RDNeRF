import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import TINY_NUMBER, HUGE_NUMBER
from collections import OrderedDict
from nerf_network import Embedder, MLPNet_T2, MLPNet
import os
import logging
# from query_depth_point.query_depth_point import QueryDepthPoint
logger = logging.getLogger(__package__)


######################################################################################
# wrapper to simplify the use of nerfnet
######################################################################################
def perturb_samples_modle(z_vals):
    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
    # uniform samples in those intervals
    t_rand = torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    return z_vals

def depth2pts_outside(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)     # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis*p_sphere, dim=-1, keepdim=True) * (1.-torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    # now calculate conventional depth
    depth_real = 1. / (depth + TINY_NUMBER) * torch.cos(theta) * ray_d_cos + d1
    return pts, depth_real

class FarDepth(nn.Module):
    def __init__(self, dim=6, D=3, W=64):
        super().__init__()
        base_layers = []
        for i in range(D):
            base_layers.append(
                nn.Sequential(nn.Linear(dim, W), nn.ReLU())
            )
            dim = W
        self.depth_head = nn.Sequential(nn.Linear(dim, 1),nn.Sigmoid())
        self.base_layers =  nn.Sequential(*base_layers)
        #self.base_layers.apply(sine_init)

    def forward(self, frustum_dists):
        far_depth_fea = self.base_layers(frustum_dists)
        fd = self.depth_head(far_depth_fea)
        return fd

class NerfNet(nn.Module):
    def __init__(self, args,m):
        super().__init__()
        self.depths=[]
        self.rgbs = []
        
        # foreground
        self.fg_embedder_position = Embedder(input_dim=3,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.fg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)
        self.fg_net = MLPNet_T2(D=args.netdepth, W=args.netwidth,
                             input_ch=self.fg_embedder_position.out_dim,
                             input_ch_viewdirs=self.fg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs,m=m)
        # self.fg_net = MLPNet(D=args.netdepth, W=args.netwidth,
        #                      input_ch=self.fg_embedder_position.out_dim,
        #                      input_ch_viewdirs=self.fg_embedder_viewdir.out_dim,
        #                      use_viewdirs=args.use_viewdirs,m=m)

        self.fd_enhance = FarDepth(dim=6,D=4,W=256)
        self.flag = 0

    def forward(self, ray_o, ray_d,fg_near_depth,fg_far_depth0, fg_z_vals,idx,is_train):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :return
        '''
        ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]                               
        viewdirs = ray_d / (ray_d_norm + TINY_NUMBER)      # [..., 3]                               
        dots_sh = list(ray_d.shape[:-1])                               
        
        inputd = torch.cat((ray_o,ray_d), 1)
        fd = self.fd_enhance(inputd)

        fg_far_depth = fd.squeeze(-1)
        fg_far_depth = fg_far_depth*9.0 + 1.0
        # fg_far_depth = fg_far_depth0.squeeze(-1)
        N_samples = fg_z_vals.shape[-1] 

        if idx == 0:
            fg_z_vals = fg_near_depth.unsqueeze(-1)*(1.-fg_z_vals) + fg_far_depth.unsqueeze(-1)  * fg_z_vals                 
            if is_train: 
                fg_z_vals = perturb_samples_modle(fg_z_vals)   # random perturbation during training                
        else:
            pass

        ######### render foreground
        dots_sh = list(ray_d.shape[:-1])
        fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        
        fg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d
        
        input = torch.cat((self.fg_embedder_position(fg_pts),
                           self.fg_embedder_viewdir(fg_viewdirs)), dim=-1)
        fg_raw = self.fg_net(input)
        # scores = fg_raw['score']

        # alpha blending
        fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
        # account for view directions
        fg_dists = ray_d_norm * torch.cat((fg_dists, fg_far_depth.unsqueeze(-1)), dim=-1)  # [..., N_samples]
        fg_alpha = 1. - torch.exp(-fg_raw['sigma'] * fg_dists)  # [..., N_samples]
        T = torch.cumprod(1. - fg_alpha + TINY_NUMBER, dim=-1)   # [..., N_samples]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]
        fg_weights = fg_alpha * T  # [..., N_samples]
        fg_rgb_map = torch.sum(fg_weights.unsqueeze(-1) * fg_raw['rgb'], dim=-2)  # [..., 3]
        fg_depth_map = torch.sum(fg_weights * fg_z_vals, dim=-1)     # [...,]

        # composite foreground
        rgb_map = fg_rgb_map
        depth_map = fg_depth_map

        # self.flag += 1
        # if self.flag == 2:
        #     print(self.flag)
        #     score = scores[0,...].cpu().float().numpy()
        #     fp = './score_test.npz'
        #     np.savez_compressed(fp, score)
        #     exit(0)

        # print(fg_z_vals.size()) 
        ret = OrderedDict([('rgb', rgb_map),            # loss
                            ('depth',depth_map),
                           ('fg_weights', fg_weights),  # importance sampling
                            ('fg_rgb', fg_rgb_map),      # below are for logging
                            ('fg_depth', fg_depth_map),
                           ('alpha',fg_alpha),
                           ('fg_z_vals',fg_z_vals),
                           ('fg_far_depth',fg_far_depth),
                           ('fg_far_depth2',fg_far_depth)
                           ]
                           )

        return ret


def remap_name(name):
    name = name.replace('.', '-')  # dot is not allowed by pytorch
    if name[-1] == '/':
        name = name[:-1]
    idx = name.rfind('/')
    for i in range(2):
        if idx >= 0:
            idx = name[:idx].rfind('/')
    return name[idx + 1:]


class NerfNetWithAutoExpo(nn.Module):
    def __init__(self, args, optim_autoexpo=False, img_names=None, m=0):
        super().__init__()
        self.nerf_net = NerfNet(args,m)
        self.optim_autoexpo = optim_autoexpo
        if self.optim_autoexpo:
            assert(img_names is not None)
            logger.info('Optimizing autoexposure!')

            self.img_names = [remap_name(x) for x in img_names]
            logger.info('\n'.join(self.img_names))
            self.autoexpo_params = nn.ParameterDict(OrderedDict([(x, nn.Parameter(torch.Tensor([0.5, 0.]))) for x in self.img_names]))

    def forward(self, ray_o, ray_d, fg_near_depth,fg_z_max, fg_z_vals, idx, is_train, img_name=None):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :return
        '''
        ret = self.nerf_net(ray_o, ray_d,fg_near_depth,fg_z_max, fg_z_vals, idx, is_train)

        if img_name is not None:
            img_name = remap_name(img_name)
        if self.optim_autoexpo and (img_name in self.autoexpo_params):
            autoexpo = self.autoexpo_params[img_name]
            scale = torch.abs(autoexpo[0]) + 0.5 # make sure scale is always positive
            shift = autoexpo[1]
            ret['autoexpo'] = (scale, shift)

        return ret
