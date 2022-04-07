import numpy as np
import torch
from torch.nn import functional as F
from nerf_encodings import sample_rays, resample_rays, integrated_pos_enc, viewdir_enc, posenc
from typing import Tuple, List
from torch import nn

class KiloNeRFContainer(nn.Module):
    def __init__(self, center, size, splits):
        super(KiloNeRFContainer, self).__init__()
        
        print("splitting into", 8**splits, "networks")
        
        self.n_divs = 2**splits
        self.step_size = size/(self.n_divs)
        self.point_norm = self.step_size
        self.bottom = nn.Parameter(center - size/2,False)

        self.steps = torch.column_stack(list(map(torch.ravel,torch.meshgrid(*[torch.linspace(0,self.n_divs-1,self.n_divs)]*3))))

        self.map_matrix = nn.Parameter(torch.tensor([4**splits,2**splits,1**splits]).float(),False)

        self.centers = nn.Parameter(self.bottom + (self.steps + 0.5) * self.step_size,False)

def get_vecs(H,W,focal):
    y,x = np.mgrid[:H,:W]
    return torch.from_numpy(np.stack([(x-W/2)/focal, -(y-H/2)/focal, -np.ones_like(x)],-1))

def get_rays(vecs,pose):
    ray_dir = torch.sum(vecs[...,None,:] * pose[:3,:3],-1)
    ray_ori = pose[:3,-1].expand(ray_dir.shape)

    return ray_ori.reshape((-1,3)),ray_dir.reshape((-1,3))

def get_params(vecs,pose):
    oris, dirs = get_rays(vecs, pose)
    viewdirs = dirs / torch.linalg.norm(dirs, axis=-1, keepdim=True)
    
    return oris, dirs, viewdirs

def get_radii(dirs):
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    dx = torch.sqrt(torch.sum((dirs[:, :-1:] - dirs[:, 1:])**2, -1))
    dx = torch.cat([dx, dx[:, -2:-1]], 1).ravel()

    return dx * 2 / np.sqrt(12)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        

@torch.jit.script
def transform_points(points, bottom, step_size:float, map_matrix, point_norm:float, centers):
    with torch.no_grad():
        indices = (torch.floor((points-bottom)/step_size)@map_matrix).long()
        relative_points = (points-centers[indices])/point_norm

        return indices, relative_points
    
@torch.jit.script   
def volumetric_rendering(rgb, density, depth_vals, vecs):
    dists = depth_vals[..., 1:] - depth_vals[..., :-1]
    delta = dists * torch.linalg.norm(vecs[..., None, :], dim=-1)
    
    density_delta = density * delta
    
    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat(
        (torch.zeros_like(density_delta[..., :1]), 
         density_delta[..., :-1].cumsum(-1)),dim=-1)
                     )
    
    weights = alpha * trans
    
    comp_rgb = (weights[..., None] * rgb).sum(-2)
    acc = weights.sum(-1)
    
    depth_centers = (depth_vals[..., :-1] + depth_vals[..., 1:])/2
    distance = (weights * depth_centers).sum(-1) / acc
    distance = torch.clip(torch.nan_to_num(distance, torch.inf), depth_vals[:, 0], depth_vals[:, -1]) 
    
    return comp_rgb, distance, acc, weights

@torch.jit.script
def partition_data(indices,x,cond):
    with torch.no_grad():
        inds, counts = torch.unique(indices,return_counts=True)

        sort_mask = indices.ravel().argsort()  
        sorted_x = x.reshape((-1,x.shape[-1]))[sort_mask]

        cond = cond[:,None].expand(-1,x.shape[1],-1).reshape((-1,cond.shape[-1])) 
        sorted_cond = cond[sort_mask]

        inds: List[int] = inds.tolist()
        offsets: List[int] = counts.tolist()

        x_splits = torch.split(sorted_x, offsets)
        cond_splits = torch.split(cond, offsets)
        
        return x_splits, cond_splits, inds, offsets, sort_mask
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
   
def sample_kilonerf(network, origins, vecs, viewdirs, radii, n_samples:int, near:int, far:int, sampling_levels:int=2,deg_point_range:Tuple[int,int] = (0,16), deg_view:int = 4, random:bool=True):
    viewdirs_enc = viewdir_enc(viewdirs, 0, deg_view)
    
    out: List[Tuple[Tensor,Tensor,Tensor]] = []

    depth_vals = torch.empty(0)
    weights = torch.empty(0)

    for i_level in range(sampling_levels):
        with torch.no_grad():
            if i_level == 0:
                depth_vals, samples = sample_rays(origins, vecs, radii, n_samples, near, far, random)
            else:
                depth_vals, samples = resample_rays(origins, vecs, radii, depth_vals, weights, 0.01, random)

            indices, rel_means = transform_points(samples[0], network.bottom, network.step_size, network.map_matrix, network.point_norm, network.centers)
            samples = (rel_means,samples[1])

            samples_enc = integrated_pos_enc(samples, deg_point_range[0], deg_point_range[1])

        raw_rgb, density_raw = network.forward(indices, samples_enc, viewdirs_enc)

        rgb = torch.sigmoid(raw_rgb)
        density = F.relu(density_raw)

        comp_rgb, distance, acc, weights = volumetric_rendering(rgb, density, depth_vals, vecs)

        out.append((comp_rgb, distance, acc))
        
    return out

