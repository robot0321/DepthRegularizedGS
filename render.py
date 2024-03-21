#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import matplotlib
import imageio
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

cmapper = matplotlib.cm.get_cmap('jet_r')

def depth_colorize_with_mask(depthlist, background=(0,0,0), dmindmax=None):
    """ depth: (H,W) - [0 ~ 1] / mask: (H,W) - [0 or 1]  -> colorized depth (H,W,3) [0 ~ 1] """
    batch, vx, vy = np.where(depthlist!=0)
    if dmindmax is None:
        valid_depth = depthlist[batch, vx, vy]
        dmin, dmax = valid_depth.min(), valid_depth.max()
    else:
        dmin, dmax = dmindmax
    
    norm_dth = np.ones_like(depthlist)*dmax # [B, H, W]
    norm_dth[batch, vx, vy] = (depthlist[batch, vx, vy]-dmin)/(dmax-dmin)
    
    final_depth = np.ones(depthlist.shape + (3,)) * np.array(background).reshape(1,1,1,3) # [B, H, W, 3]
    final_depth[batch, vx, vy] = cmapper(norm_dth)[batch,vx,vy,:3]

    return final_depth


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    gt_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gt_depth_path, exist_ok=True)

    depthlist, gtdepthlist = [], []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        gt = view.original_image[0:3, :, :]
        gt_depth = view.original_depth.detach().cpu().numpy()
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        tmp_depth = np.zeros_like(gt_depth)
        vvy, vvx = np.where(gt_depth!=0)
        if len(vvy) < gt_depth.size * 0.1:
            for x in range(-2,3):
                for y in range(-2,3):
                    tmp_depth[(vvy+y).clip(0,tmp_depth.shape[0]-1), (vvx+x).clip(0,tmp_depth.shape[1]-1)] = gt_depth[vvy,vvx]
            gt_depth = tmp_depth
        
        if "depth" in results:
            # depthlist.append((results["depth"]*(results["acc"]>0.9)).detach().cpu().numpy())
            depthlist.append((results["depth"]*(results["depth"]!=0)).detach().cpu().numpy())
            gtdepthlist.append((gt_depth*(gt_depth!=0)))
    depthlist = np.concatenate(depthlist, axis=0)
    gtdepthlist = np.stack(gtdepthlist, axis=0)
    gtdmindmax = (gtdepthlist.min(), gtdepthlist.max())
    if "depth" in results:    
        # import pdb; pdb.set_trace()       
        # colorized_depth = depth_colorize_with_mask(depthlist, background.detach().cpu().numpy(), dmindmax=gtdmindmax)
        # colorized_gtdepth = depth_colorize_with_mask(gtdepthlist, background.detach().cpu().numpy(), dmindmax=gtdmindmax)
        # for idx in tqdm(range(len(colorized_depth)), desc="Rendering Depth progress"):
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            # imageio.imwrite(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), np.round(colorized_depth[idx]*255.).astype(np.uint8))
            np.save(os.path.join(depth_path, view.image_name + ".npy"), depthlist[idx])
            # imageio.imwrite(os.path.join(gt_depth_path, '{0:05d}'.format(idx) + ".png"), np.round(colorized_gtdepth[idx]*255.).astype(np.uint8))
            np.save(os.path.join(gt_depth_path, view.image_name + ".npy"), gtdepthlist[idx])


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)