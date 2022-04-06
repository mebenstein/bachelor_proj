import torch
from nerf.utils import *

def get_patched_rays(c2w, intrinsics, H, W, n_patches=512,patch_dim=3):
    # copied from torch-ngp
    device = c2w.device
    rays_o = c2w[..., :3, 3] # [B, 3]
    prefix = c2w.shape[:-2]

    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij') # for torch < 1.10, should remove indexing='ij'
    i = i.t().reshape([*[1]*len(prefix), H*W]).expand([*prefix, H*W])
    j = j.t().reshape([*[1]*len(prefix), H*W]).expand([*prefix, H*W])

    y,x = torch.randint(0,H-patch_dim,(n_patches,),device=device),torch.randint(0,W-patch_dim,(n_patches,),device=device)
    select_hs, select_ws = torch.meshgrid((torch.linspace(0, patch_dim-1, patch_dim, device=device).long(),torch.linspace(0, patch_dim-1, patch_dim, device=device).long()))
    select_inds = (y+select_hs.ravel()[:,None]).T.ravel() * W + (x+select_ws.ravel()[:,None]).T.ravel()
    select_inds = select_inds.expand([*prefix, n_patches*patch_dim**2])
    i = torch.gather(i, -1, select_inds)
    j = torch.gather(j, -1, select_inds)

    pixel_points_cam = lift(i, j, torch.ones_like(i), intrinsics=intrinsics)
    pixel_points_cam = pixel_points_cam.transpose(-1, -2)

    world_coords = torch.bmm(c2w,pixel_points_cam).transpose(-1, -2)[..., :3]
    
    rays_d = world_coords - rays_o[..., None, :]
    rays_d = F.normalize(rays_d, dim=-1)

    rays_o = rays_o[..., None, :].expand_as(rays_d)

    return rays_o, rays_d, select_inds

@torch.jit.script
def project_to_3d(inds, depth, W:int, fx:float, fy:float, cx:float, cy:float):
    """Projects points from flat image space to 3D camera reference space
       
       Args:
           inds  (1D tensor) of pixel indices 
           depth (1D tensor) of predicted depth values
           W     (int)       Image width
           fx    (float)     focal length in x-direction
           fy    (float)     focal length in y-direction
           cx    (float)     camera principle point in x-direction
           cy    (float)     camera principle point in y-direction
       Returns:
           3D tensor: 3d points 
    """
    patch_inds = inds.reshape((-1,9)) # group by local image patches

    u = patch_inds%W 
    v = patch_inds//W
    d = depth.reshape((-1,9))
    
    x_over_z = (cx - u) / fx
    y_over_z = (cy - v) / fy
    z = d / torch.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z

    return torch.stack((x,y,z),-1)

@torch.jit.script
def get_normal_matrix(points): 
    """Generate normal vector matrix for 3x3 point patches
       
       Args:
           points  (3D tensor)
       Returns:
           (n,3,3,3) tensor: list of 3x3 matrix of normal vectors 
    """
    centers = points[:,4] # get central point
    outers = points[:,[3,0,1,2,5,8,7,6,3]] # get surrounding points in clockwise order

    # calculate normal vectors
    vecs = outers - centers[:,None] 
    norms = torch.cross(vecs[:,:-1],vecs[:,1:])
    norm = norms.mean(1)

    # generate normal vector matrix
    A = torch.empty((len(norm),9,3)).cuda()
    A[:,[0,1,2,5,7,8,6,3]] = norms
    A[:,4] = norm
    
    # normalize matrix
    A = A/torch.norm(A,dim=-1)[...,None]
    
    return A.reshape((-1,3,3,3))

@torch.jit.script
def rowwise_dot(a,b):
    # row-wise dot product
    return (a*b).sum(-1)

@torch.jit.script
def rowwise_norm(a):
    # row-wise norm
    return torch.sqrt((a**2).sum(-1))

@torch.jit.script
def sine_distance(v1,v2):
    # sine distance function
    return 1-(rowwise_dot(v1,v2)/(v1.norm(2,-1)*v2.norm(2,-1)))**2

@torch.jit.script
def xy_grad_patches(a):
    # helper function to return matrix with differnt offsets to calculate gradients 
    return a[:, :-1, :-1],\
           a[:, :-1, 1:],\
           a[:, 1:, :-1]
        
@torch.jit.script
def xy_grad(a):
    # calculate gradient of matrix "a" in x and y direction 
    a00,a01,a10 = xy_grad_patches(a)
    
    return (a00-a01)**2, (a00-a10)**2
        
@torch.jit.script
def get_loss_3dgs(depth_patches, image_patches, A):    
    """Calculaet the 3dgs loss
       
       Args:
           depth_patches  (3x3 depth values)
           image_patches  (3x3x3 radiance values)
           A              (nx3x3x3 normal vectors)
       Returns:
           float: 3dgs loss
    """
    
    dxD,dyD = xy_grad(depth_patches)  # gradient of depth values
    dxI, dyI = xy_grad(image_patches) # gradient of radiance values 
    
    A00,A01,A10 = xy_grad_patches(A) # gradient of normal vectors
    sxA = sine_distance(A00,A01) # sine distance of normal vectors in x-direction
    syA = sine_distance(A00,A10) # sine distance of normal vectors in y-direction

    edxI = torch.norm(torch.exp(-dxI).reshape((-1,12)),dim=1)[:,None,None] # scaling factor in x-direction
    edyI = torch.norm(torch.exp(-dyI).reshape((-1,12)),dim=1)[:,None,None] # scaling factor in y-direction

    dgs_loss = edxI * (sxA + dxD) + edyI * (syA + dyD) # final equation
    
    return torch.nan_to_num(dgs_loss,1.0,1.0).mean()

@torch.jit.script
def loss_3dgs(inds, depth_patches, image_patches, W:int, fx:float, fy:float, cx:float, cy:float):
    # Helper function to combine steps
    points = project_to_3d(inds,depth_patches, W, fx, fy, cx, cy)
    A = get_normal_matrix(points)
    return get_loss_3dgs(depth_patches, image_patches, A)