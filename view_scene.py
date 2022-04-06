import sys
import os
sys.path.insert(0,os.getcwd()+"/torch-ngp")

from nerf.network_tcnn import NeRFNetwork
from nerf.provider import NeRFDataset
from easydict import EasyDict
from nerf.gui import NeRFGUI
from nerf.utils import *
import torch

def main(workspace:str, scale:float, bound: float):
    """Train a NeRF on a scene and save the weights to a workspace

    Args:
        seq_name (str): name of the sequence in the data folder
    """    
    model = NeRFNetwork(
        encoding="hashgrid", encoding_dir="sphere_harmonics", 
        num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64, 
        cuda_ray=True,
    ).eval()

    opt = EasyDict()
    opt.gui = True
    opt.mode = "colmap"
    opt.path = "data/lego"
    opt.W = 800
    opt.H = 800
    opt.radius=5
    opt.test = True
    opt.max_spp = 64
    opt.scale = scale
    opt.bound = bound

    trainer = Trainer('ngp', vars(opt), model, workspace="workspace/"+workspace, optimizer=None, criterion=None, ema_decay=0.95, fp16=True, lr_scheduler=None,eval_interval=10)

    train_dataset = NeRFDataset(opt.path, type='all', mode=opt.mode, scale=opt.scale, preload=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=False)
    trainer.train_loader = train_loader


    gui = NeRFGUI(opt, trainer)
    gui.render()

if __name__ == "__main__":
    main("lego",0.33,3)