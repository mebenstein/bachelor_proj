# Most methods here were adapted from the MiP-NeRF implementation

import numpy as np
import torch
from typing import Tuple, List

@torch.jit.script   
def viewdir_enc(x, min_deg:int, max_deg:int, append_identity:bool=True):
    """The positional encoding used by the original NeRF paper."""
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)], device=x.device, dtype=x.dtype)
    xb = torch.reshape((x[..., None, :] * scales[:, None]),
                   list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
    
    if append_identity:
        return torch.cat([x,four_feat], dim=-1)
    else:
        return four_feat
 
@torch.jit.script      
def expected_sin(x, x_var):
    return torch.exp(-0.5 * x_var) * torch.sin(x)

@torch.jit.script    
def integrated_pos_enc(x:Tuple[torch.Tensor,torch.Tensor], min_deg:int, max_deg:int):
    """get MiP encoding

    Args:
        x (Tuple[torch.Tensor,torch.Tensor]): sample point and cone information
        min_deg (int): cone min span
        max_deg (int): cone max span

    Returns:
        tensor: IPE vector
    """    
    x, x_cov = x
    
    scales = 2**torch.linspace(min_deg, max_deg, (max_deg - min_deg), device=x.device, dtype=x.dtype)   
        
    shape = list(x.shape[:-1]) + [-1]

    y = torch.reshape(x[..., None, :] * scales[:, None], shape)
    y_var = torch.reshape(x_cov[..., None, :] * scales[:, None]**2,  shape)
    
    v = torch.exp(-0.5*y_var)
    
    return torch.cat((v*torch.sin(y),v*torch.cos(y)),-1)#expected_sin(torch.cat((y, y + 0.5 * torch.pi),-1), torch.cat((y_var, y_var),-1))
    
@torch.jit.script   
def lift_gaussian(vecs, depth_mean, depth_var, radius_var, diag:bool):
    # see original MiP-NeRF implementation
    mean = vecs[..., None, :] * depth_mean[..., None]
    vec_mag_sq = torch.maximum(torch.tensor(1e-10,device=vecs.device, dtype=vecs.dtype), torch.sum(vecs**2, dim=-1))[:,None]
    
    if diag:
        vec_outer = vecs**2
        null_outer = 1 - vec_outer / vec_mag_sq
        depth_cov = depth_var[..., None]
        xy_cov = radius_var[..., None]
    else:
        vec_outer = vecs[..., None] * vecs[..., None, :]
        eye = torch.eye(vecs.shape[-1])
        null_outer = eye - depth_var[..., None] * (vecs / vec_mag_sq)[..., None, :]
        depth_cov = depth_var[..., None, None]
        xy_cov = depth_var[..., None, None]
        
    depth_cov = depth_cov * vec_outer[..., None, :]
    xy_cov = xy_cov * null_outer[..., None, :]
    cov = depth_cov + xy_cov
    
    return mean, cov

@torch.jit.script   
def generate_conical_frustum_gaussian(vecs, depth_start, depth_end, radius, diag:bool):
    # see original MiP-NeRF implementation
    mu = (depth_start + depth_end) / 2
    musq = mu**2

    hwsq = (depth_end - mu)**2
    hwft = hwsq**2
    
    depth_mean = mu + (2 * mu * hwsq) / (3 * musq + hwsq)
    depth_var = hwsq / 3 - (4 / 15) * ((hwft * (12 * musq - hwsq)) / (3 * musq + hwsq)**2)
    radius_var = radius[:,None]**2 * (musq / 4 + (5 / 12) * hwsq - (4 / 15) * hwft / (3 * musq + hwsq))
   
    return lift_gaussian(vecs, depth_mean, depth_var, radius_var, diag)
  
@torch.jit.script     
def cast_rays(depth_values, origins, vecs, radii, diag:bool=True):
    # see original MiP-NeRF implementation
    depth_start = depth_values[...,:-1]
    depth_end = depth_values[...,1:]
    
    means, covs = generate_conical_frustum_gaussian(vecs, depth_start, depth_end, radii, diag)
    means += origins[..., None, :]
    
    return means, covs

@torch.jit.script   
def get_sample_depths(vecs, n_samples:int, near:int, far:int, random:bool=True):
    # see original MiP-NeRF implementation
    depth_vals = torch.linspace(near, far, n_samples + 1,  device=vecs.device, dtype=vecs.dtype)
    
    if random:
        rand_offset = (far - near) / n_samples * (torch.rand((vecs.shape[0], n_samples + 1), device=vecs.device, dtype=vecs.dtype) - 0.5)
        depth_vals = depth_vals + rand_offset
    else:
        depth_vals = torch.broadcast_to(depth_vals, (vecs.shape[0], n_samples + 1))
        
    return depth_vals

@torch.jit.script   
def sample_rays(origins, vecs, radii,  n_samples:int, near:int, far:int, random:bool=True):
    # see original MiP-NeRF implementation
    depth_vals = get_sample_depths(vecs, n_samples, near, far, random)
    means, covs = cast_rays(depth_vals, origins, vecs, radii)
    
    return depth_vals, (means, covs)
    
@torch.jit.script   
def find_interval(mask,x):
    # see original MiP-NeRF implementation
    return torch.where(mask, x[..., None], x[..., :1, None]).max(-2)[0], \
           torch.where(~mask, x[..., None], x[..., -1:, None]).min(-2)[0]
  
@torch.jit.script
def _get_tensor_eps( # helper function to get dtype eps
    x: torch.Tensor,
    epsbf16: float = torch.finfo(torch.bfloat16).eps,
    eps16: float = torch.finfo(torch.float16).eps,
    eps32: float = torch.finfo(torch.float32).eps,
    eps64: float = torch.finfo(torch.float64).eps,
) -> float:
    if x.dtype == torch.float16:
        return eps16
    elif x.dtype == torch.bfloat16:
        return epsbf16
    elif x.dtype == torch.float32:
        return eps32
    elif x.dtype == torch.float64:
        return eps64
    else:
        raise RuntimeError(f"Expected x to be floating-point, got {x.dtype}")
    
@torch.jit.script     
def sorted_piecewise_constant_pdf(bins, weights, random:bool):
    # see original MiP-NeRF implementation
    weight_sum = weights.sum(-1, keepdim=True)
    padding = torch.clamp(1e-5 - weight_sum, min=0)
    
    weights += padding/weights.shape[-1]
    weight_sum += padding
        
    pdf = weights/weight_sum
    cdf = torch.clamp(torch.cumsum(pdf[...,:-1],-1),max=1)
    
    shape = list(cdf.shape)
    shape[-1] = 1
    cdf = torch.cat([torch.zeros(shape, device=bins.device, dtype=bins.dtype), cdf, torch.ones(shape, device=bins.device, dtype=bins.dtype)], -1)
    
    n = bins.shape[1]-1 
    shape[-1] = n
    
    if random:
        s = 1/n
        u = torch.linspace(0, n, n, device=bins.device, dtype=bins.dtype) + torch.rand(shape, device=bins.device, dtype=bins.dtype) * (s - _get_tensor_eps(bins))
        u = torch.clamp(u,max=1-_get_tensor_eps(bins))
    else:
        u = torch.broadcast_to(torch.linspace(0,1 - _get_tensor_eps(bins), n, device=bins.device, dtype=bins.dtype), shape)
    
    mask = u[..., None, :] >= cdf[..., :, None]
    
    bins_g0, bins_g1 = find_interval(mask, bins)
    cdf_g0, cdf_g1 = find_interval(mask, cdf)
    
    t = torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0.0)
    t = torch.clip(t, 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    #print((bins_g0[..., 1:] - bins_g0[..., :-1])[5:])
    return samples
    

@torch.jit.script   
def resample_rays(origins, vecs, radii, depth_vals, weights, padding:float, random:bool=True):
    # see original MiP-NeRF implementation
    weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], -1)
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = (weights_max[...,:-1] + weights_max[...,1:])/2
    
    weights = weights_blur + padding
    
    depth_vals = sorted_piecewise_constant_pdf(depth_vals, weights, random)
    
    means, covs = cast_rays(depth_vals, origins, vecs, radii)
    
    return depth_vals, (means, covs)
    
@torch.jit.script   
def cumprod_exclusive(tensor,dim:int=-1):
    # implementation of exclusive cumprod
    cumprod = torch.cumprod(tensor, dim)
    cumprod = torch.roll(cumprod, 1, dim)
    cumprod[..., 0] = 1.0
    return cumprod
    
@torch.jit.script   
def posenc(x):
    rets = [x]
    for i in range(6):
            rets.append(torch.sin(2.**i * x))
            rets.append(torch.cos(2.**i * x))
    return torch.cat(rets, -1)
    