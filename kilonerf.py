from kilonerf_helpers import *
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from typing import Tuple, List

import click
import time

class NeRFModule(nn.Module):
    def __init__(self,in_d,dir_d,w):
        super(NeRFModule, self).__init__()
        
        self.a = nn.Linear(in_d,w)
        self.b = nn.Linear(w,w)
        self.c = nn.Linear(w,w)     # output density
        self.d = nn.Linear(w+dir_d,w) # inject view direction
        self.e = nn.Linear(w,3)       # output color
        
    def forward(self, x, cond) -> torch.Tensor:      
        x = F.relu(self.a(x))
        x = F.relu(self.b(x))
        x = self.c(x)
        
        sigma = F.relu(x[...,:1])
        
        x = torch.cat((x,cond),-1)
        x = F.relu(self.d(x))
        
        rgb = torch.sigmoid(self.e(x))
        
        return torch.concat((rgb, sigma),-1)
    
class NeRF(KiloNeRFContainer):
    def __init__(self,center, size, splits):
        super(NeRF, self).__init__(center, size, splits)    
        
        self.networks = nn.ModuleList([NeRFModule(96, 27, 32) for x in range(len(self.centers))])
    
    def forward(self, indices, x, cond):
        x_splits,cond_splits,inds,offsets,sort_mask = partition_data(indices,x,cond)
            
        offset = 0
        outs: List[Future[Tensor]] = []
            
        for i,net in enumerate(self.networks):
            if i in inds:
                outs.append(torch.jit.fork(net, x_splits[offset], cond_splits[offset]))
                offset += 1
        
        data = torch.empty(x.shape[0]*x.shape[1],4, dtype=x.dtype,device=x.device)
         
        offset = 0
        for i,f in enumerate(outs):
            temp = offset + offsets[i]
            data[offset:temp] = torch.jit.wait(f)
            offset = temp
            
        data = data[sort_mask.argsort()].reshape((x.shape[0],x.shape[1],4))
            
        return data[...,:3], data[...,3]
    
@click.command()
@click.argument('num_iterations',required=True,type=int)
@click.argument('parallell',required=True,type=bool)
def main(num_iterations, parallell):
    torch.set_num_interop_threads(32)

    dev = "cuda"
    dtype = "torch.BFloat16Tensor"

    data = np.load('data/tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    H, W = 100,100

    i_img = 101 

    pose = torch.from_numpy(poses[i_img]).type(dtype).to(dev)
    target = torch.from_numpy(images[i_img]).type(dtype).cuda().reshape((-1,3))

    vecs = get_vecs(H,W,focal).type(dtype).to(dev)
    oris, dirs, view_dirs = get_params(vecs,pose) 
    radii = get_radii(vecs)

    model = NeRF(torch.zeros(3),10,4).type(dtype).cuda()

    if parallell:
        net = torch.jit.script(model)
    else:
        net = model

    net.apply(init_weights)
    optim = torch.optim.Adam(net.parameters(), lr=5e-4)

    with torch.cuda.amp.autocast():
        s = time.time()
        for i in range(num_iterations):
            net.zero_grad()

            res = list(sample_kilonerf(net,oris, dirs, view_dirs, radii, 64, 2, 6))

            crgb, cdist, cacc = res[0]
            rgb, dist, acc = res[1]

            loss = 0.1 * ((crgb - target)**2).mean()
            loss += ((rgb - target)**2).mean()
            loss.backward()
        
        print("Average time per iteration:",(time.time() - s)/num_iterations)

if __name__ == "__main__":
    main()