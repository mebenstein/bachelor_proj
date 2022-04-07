import sys
import os
sys.path.insert(0,os.getcwd()+"/torch-ngp")

from nerf.network_tcnn import NeRFNetwork
from nerf.provider import NeRFDataset
from easydict import EasyDict
from nerf.gui import NeRFGUI
from nerf.utils import *
import torch

import click

@click.command()
@click.argument('workspace',required=True)
@click.option('--bound',default=3,type=int,help="scene bounding box size")
@click.option('--scale',default=0.33,type=int,help="pose scale adjustment")
def main(workspace:str, scale:float, bound: float):
    """View a learned representation

    Args:
        workspace (str): workspace name\n
        scale (float): pose scale adjustment\n
        bound (float): scene bounding box size\n
    """    

    model = NeRFNetwork(
        encoding="hashgrid", encoding_dir="sphere_harmonics", 
        num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64, 
        cuda_ray=True,
    ).eval()

    opt = EasyDict()
    opt.gui = False
    opt.mode = "colmap"
    opt.path = "data/desk_concave"
    opt.W = 800
    opt.H = 800
    opt.test = True
    opt.scale = scale
    opt.bound = bound

    train_dataset = NeRFDataset(opt.path, type='train', mode=opt.mode, scale=opt.scale, preload=False)
    trainer = Trainer('ngp', vars(opt), model, workspace="workspace/"+workspace, optimizer=None, criterion=None, ema_decay=0.95, fp16=True, lr_scheduler=None)

    sample = train_dataset[0]
    pose = sample["pose"]
    intr = sample["intrinsic"]
    focal = intr[0,0].item()
    H,W = sample["image"].shape[-3:-1]

    fig, plots = plt.subplots(1,2,figsize=(40,20))
    res = trainer.test_gui(pose,intr,W,H)

    print("depth range", res["depth"].min().item(),res["depth"].max().item())

    plots[0].matshow(res["image"])
    plots[1].matshow(res["depth"])
    plt.show()

if __name__ == "__main__":
    main("desk_concave",0.33,3)