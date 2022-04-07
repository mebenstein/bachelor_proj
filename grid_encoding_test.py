import cv2
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from typing import Tuple, List

from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

import matplotlib.pyplot as plt
import click

dev = "cuda"
dtype = "torch.FloatTensor"

def get_mapping(coords,level):
    """simple alternative to hashing function

    Args:
        coords (tensor): 2D coordinates
        level (int): level to get the voxel indices for

    Returns:
        4D tensor: surrounding voxel indices
    """    
    k = (coords//(np.array([H,W])//level)*[1,level]).astype(int).sum(-1)
    k = np.minimum(level**2-1,k)
    idx = k+k//level
    
    return np.column_stack((idx,idx+1,idx+l+1,idx+l+2))

class NetworkModule(nn.Module):
    def __init__(self,in_d,w):
        """simple module that learns a mapping from n-D space to rgb

        Args:
            in_d (int): input dimension
            w (int): network width
        """        
        super(NetworkModule, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_d,w),
            nn.ReLU(),
            nn.Linear(w,w),
            nn.ReLU(),
            nn.Linear(w,w),
            nn.ReLU(),
            nn.Linear(w,w),
            nn.ReLU(),
            nn.Linear(w,w),
            nn.ReLU(),
            nn.Linear(w,w),
            nn.ReLU(),
            nn.Linear(w,w),
            nn.ReLU(),
            nn.Linear(w,3),
            nn.Sigmoid()
        )

    def forward(self, x) -> torch.Tensor:      
        return self.layers(x)


@click.command()
@click.argument('path',required=True)
@click.argument('n',required=True,type=int)
def main(path:str,n:int):
    """Run simple hash encoding example that learns to represent a single image

    Args:\n
        path (str): image path\n
        n (int): number of iterations to train for\n
    """    
    global H,W

    # load image
    img =  torch.from_numpy(cv2.imread(path)[...,[2,1,0]]).cuda()/255

    # define encoding parameters
    L = 32
    c_F = 2
    H,W = img.shape[:-1]

    # encoding variables
    params = [torch.rand(((3+i)**2,c_F),dtype=torch.bfloat16,requires_grad=True,device=dev) for i in range(L)]
    
    # calculate the voxel coordinates for each variable
    param_coords = [torch.from_numpy(np.column_stack(map(np.ravel,np.mgrid[:3+i,:3+i]))*[H,W]//(3+i)).type(dtype).cuda() for i in range(L)]    
    
    # calculate the coordinates of all points
    coords = torch.from_numpy(np.column_stack(map(np.ravel,np.mgrid[:H,:W]))).type(dtype).cuda()

    # determine the surrounding voxels for each point
    mappings = [torch.from_numpy(get_mapping(coords.cpu().float().numpy(),2+l)).cuda() for l in range(L)]
    
    # determine relative positioning weight for each point
    weights = [1/(torch.sum(param_coords[i][mappings[i]] - coords[:,None],axis=-1)+0.1) for i in range(L)]

    # initiate model
    model = NetworkModule(c_F*L,64).type(dtype).cuda()

    # optimizer with weight decay for model weights only
    optim = torch.optim.Adam([
        {'name': 'encoding', 'params': params},
        {'name': 'net', 'params': list(model.parameters()), 'weight_decay': 1e-6},
    ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    ema = ExponentialMovingAverage(model.parameters(), decay=0.95)

    fix,axs = plt.subplots(n//10,figsize=(20,20))

    bar = tqdm(range(n),desc="Steps")

    for k in bar:
        optim.zero_grad()

        # gather all parameters 
        inps = []
        for i in range(L):
            mapping = mappings[i]
            inps.append((params[i][mapping] * weights[i][...,None]).sum(1))

        # concatinate them to from the input encoding
        inps = torch.cat(inps,-1)

        # forward it to the model
        rgb = model(inps).reshape((H,W,3))
        loss = ((rgb-img)**2).mean()

        loss.backward()
        optim.step()

        bar.set_description("Epochs (mean loss {}):".format(loss.item()))

        if k % 10 == 0:
            ema.update()

            ax = axs[k//10]
            ax.imshow(rgb.detach().cpu().float().numpy().reshape((H,W,3)))
            ax.set_yticks([])
            ax.set_xticks([])
                
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()