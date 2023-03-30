import torch
# import torch.nn as nn
import torch.optim
import torch.distributed
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import numpy as np
import os
# from collections import OrderedDict
# from ddp_model import NerfNet
import time
from data_loader_split import load_data_split
from utils import mse2psnr, colorize_np, to8b
import imageio
from ddp_train_nerf import config_parser, setup_logger, setup, cleanup, render_single_image, create_nerf
import logging
import cv2
# import line_profiler
from matplotlib.cm import get_cmap
from PIL import Image

CM_MAGMA = (np.array([get_cmap('magma').colors]).
            transpose([1, 0, 2]) * 255)[..., ::-1].astype(np.uint8)

def visualize_depth(depth, depth_min=None, depth_max=None):
    """Visualize the depth map with colormap.
    Rescales the values so that depth_min and depth_max map to 0 and 1,
    respectively.
    """
    if depth_min is None:
        depth_min = np.amin(depth)

    if depth_max is None:
        depth_max = np.amax(depth)

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled = depth_scaled ** 0.5
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)

    return ((cv2.applyColorMap(
        depth_scaled_uint8, CM_MAGMA) / 255) ** 2.2) * 255

logger = logging.getLogger(__package__)
def ddp_test_nerf(rank, args):
    ###### set up multi-processing
    setup(rank, args.world_size)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()
    timesum= [0,0,0]
  
    ###### decide chunk size according to gpu memory
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 4096

    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf(rank, args)

    render_splits = ['test']
    # start testing
    psnr_sum=0
    for split in render_splits:
        out_dir = os.path.join(args.basedir, args.expname,
                               'render_{}_{:06d}'.format(split, start))
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
        print(out_dir)
        
        ###### load data and create ray samplers; each process should do this
        split="test"
        ray_samplers = load_data_split(args.datadir, args.scene, split, try_load_min_depth=args.load_min_depth, skip=1, test_flag=True)
        time_sum=0.0
        for idx in range(len(ray_samplers)):
            ### each process should do this; but only main process merges the results
            fname = '{:06d}.png'.format(idx)
            if ray_samplers[idx].img_path is not None:
                fname = os.path.basename(ray_samplers[idx].img_path)

            if os.path.isfile(os.path.join(out_dir, fname)):
                logger.info('Skipping {}'.format(fname))
                continue
  
            time0 = time.time()
            ret= render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size)
            dt = time.time() - time0
  
            rgbs = []
            depths = []
            if rank == 0:    # only main process should do this
                logger.info('Rendered {} in {} seconds'.format(fname, dt))
                time_sum = time_sum+dt
                # only save last level
                im = ret[-1]['rgb'].numpy()

                im = to8b(im)
                rgbs.append(im)
                imageio.imwrite(os.path.join(out_dir, fname), im)

                # depth = ret[-1]['fg_far_depth'].numpy()
                # depth = depth*9.0/depth.max() + 1.0
                # depth = (depth*6000).astype(np.uint16)
                # depth0 = (depth0*6000).astype(np.uint16)
                # fnamed = fname.replace('.jpg','.png')
                # cv2.imwrite(os.path.join(out_dird, 'depth_' + fnamed), depth)


            torch.cuda.empty_cache()
    print("------------")
    print(psnr_sum/len(ray_samplers))
    print(time_sum/len(ray_samplers))
    # clean up for multi-processing
    cleanup()


def test():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    torch.multiprocessing.spawn(ddp_test_nerf,
                                args=(args,),
                                nprocs=args.world_size,
                                join=True)


if __name__ == '__main__':
    setup_logger()
    test()

