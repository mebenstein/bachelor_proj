import sys
import os
sys.path.insert(0,os.getcwd()+"/torch-ngp")

import torch
from nerf.network_tcnn import NeRFNetwork
from nerf.provider import NeRFDataset
from nerf.utils import *
from easydict import EasyDict

import numpy as np
from helpers import *

import click

@click.command()
@click.argument('seq_name',required=True)
@click.argument('loss_type',required=True,type=click.Choice(["smooth","normal","rgb"]))
@click.option('--epochs',default=60,type=int,help="number of epochs to train")
@click.option('--bound',default=3,type=int,help="scene bounding box size")
@click.option('--scale',default=0.33,type=int,help="pose scale adjustment")
@click.option('--num_ray_patches',default=512,type=int,help="number of 3x3 sample patches")
@click.option('--workspace',default="default",type=str,help="workspace name")
def main(seq_name:str, loss_type:str, epochs:int, bound:float, scale:float, num_ray_patches:int, workspace:str):
    """Train a NeRF on a scene and save the weights to a workspace

    Args:\n
        seq_name (str): name of the sequence in the data folder\n
        loss_type (str): type of loss, either "smooth", "normal" or "rgb"\n
        epochs (int): number of epochs to train\n
        bound (float): scene bounding box size\n
        scale (float): pose scale adjustment\n
        num_ray_patches (int): number of 3x3 sample patches\n
        workspace (str): workspace name\n
    """    
    model = NeRFNetwork(
        encoding="hashgrid", encoding_dir="sphere_harmonics", 
        num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64, 
        cuda_ray=True,
    )

    criterion = torch.nn.HuberLoss(delta=0.1).cuda()

    optimizer = lambda model: torch.optim.Adam([
        {'name': 'encoding', 'params': list(model.encoder.parameters())},
        {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
    ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    opt = EasyDict()
    opt.gui = False
    opt.path = "data/"+seq_name
    opt.mode = "colmap"
    opt.scale = scale
    opt.bound = bound
    opt.num_rays = num_ray_patches*9

    train_dataset = NeRFDataset(opt.path, type='train', mode=opt.mode, scale=opt.scale, preload=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.33)
    trainer = Trainer('ngp', vars(opt), model, workspace="workspace/"+workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=True, lr_scheduler=scheduler)

    bg_color = torch.ones(3, device="cuda") # [3], fixed white background
    bar = tqdm.tqdm(range(trainer.epoch,epochs),initial=trainer.epoch,total=epochs,desc="Epochs")

    for epoch in bar:
        torch.cuda.empty_cache()

        with torch.cuda.amp.autocast():
            trainer.model.update_extra_state(bound)
        losses = 0
        
        for data in tqdm.tqdm(train_loader,desc="Samples",leave=False):
            data = trainer.prepare_data(data)

            trainer.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                #preds, truths, loss = trainer.train_step(data)
                images = data["image"][...,:3] # [B, H, W, 3/4]
                poses = data["pose"] # [B, 4, 4]
                intrinsics = data["intrinsic"] # [B, 3, 3]

                # sample rays 
                B, H, W, C = images.shape
                rays_o, rays_d, inds = get_patched_rays(poses, intrinsics, H, W, num_ray_patches)#my_rays(poses, intrinsics, H, W, 4608)
                images = torch.gather(images.reshape(B, -1, C), 1, torch.stack(C*[inds], -1)) # [B, N, 3/4]
                
                gt_rgb = images.reshape((-1,3,3,3))

                outputs = trainer.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, bound=bound)
                pred_rgb = outputs['rgb'].reshape((-1,3,3,3))
                pred_depth = outputs['depth'].reshape((-1,3,3))+1e-6

                rgb_loss = trainer.criterion(pred_rgb, gt_rgb)

                if loss_type == "smooth":
                    v00 = pred_depth[:, :-1, :-1]
                    v01 = pred_depth[:, :-1, 1:]
                    v10 = pred_depth[:, 1:, :-1]
                    depth_loss = ((v00 - v01) ** 2 + (v00 - v10) ** 2).mean()
                    loss = rgb_loss + 1e-2 * depth_loss
                elif loss_type == "normal":
                    dgs_loss = loss_3dgs(inds,pred_depth,pred_rgb, W, fx, fy, cx, cy)
                    loss = rgb_loss + 1e-4 * dgs_loss
                else:
                    loss = rgb_loss

                preds, truths, loss = pred_rgb, gt_rgb, loss
                
            trainer.scaler.scale(loss).backward()
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()

            loss_val = loss.item()
            losses += loss_val/len(train_loader)

        bar.set_description("Epochs (mean loss {}):".format(losses))
        
        trainer.ema.update()
        trainer.lr_scheduler.step()

        if epoch % 10 == 0 and epoch > 0:
            trainer.save_checkpoint(full=True)
            trainer.epoch += 1

if __name__ == "__main__":
    main()