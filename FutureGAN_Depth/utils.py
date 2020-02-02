# =============================================================================
# Utils
# =============================================================================

import torch
from torchvision.utils import make_grid
import numpy as np

# =============================================================================
# generate a grid of video frames

def make_video_grid(x):
 
    x = x.clone().cpu()
    grid = torch.FloatTensor(x.size(0)*x.size(2), x.size(1), x.size(3), x.size(4)).fill_(1)
    k = 0
    for i in range(x.size(0)):
        for j in range(x.size(2)):            
            grid[k].copy_(x[i,:,j,:,:])
            k = k+1
    grid = make_grid(grid, nrow=x.size(2), padding=0, normalize=True, scale_each=False)
    return grid


# =============================================================================
# save a grid of video frames
    
def save_video_grid(x, path, imsize=512):
    
    from PIL import Image    
    grid = make_video_grid(x)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    if x.size(2)<x.size(0):
        imsize_ratio = int(x.size(0)/x.size(2))
        im = im.resize((imsize,int(imsize*imsize_ratio)), Image.NEAREST)   
    else:
        imsize_ratio = int(x.size(2)/x.size(0))
        im = im.resize((int(imsize*imsize_ratio), imsize), Image.NEAREST)      
    im.save(path)


# =============================================================================
# generate a grid of video frames

def make_depth_grid(x):
    x = x.clone().cpu()
    grid = torch.FloatTensor(x.size(0) * x.size(2), x.size(1), x.size(3), x.size(4)).fill_(1)
    #print(grid.shape)
    k = 0
    for i in range(x.size(0)):
        for j in range(x.size(2)):
            grid[k].copy_(x[i, :, j, :, :])
            k = k + 1
    #print(grid.shape)
    grid = make_grid(grid, nrow=x.size(2), padding=0, normalize=False, scale_each=False)
    return grid[0,:,:] #make_grid automatically copies a 1-channel image to 3 channels, so we only want the first channel here


# =============================================================================
# save a grid of depth video frames

def save_depth_grid(x, path, imsize=512):
    from PIL import Image
    grid = make_depth_grid(x)
    #ndarr = np.array(grid.mul(10000).clamp(0, 10000).to(torch.int32).permute(1, 2, 0).numpy(),dtype='int32')
    ndarr = np.array(grid.add(1).mul(5000).clamp(0, 10000).to(torch.int32).numpy(), dtype='int32')
    im = Image.fromarray(ndarr,"I")
    if x.size(2) < x.size(0):
        imsize_ratio = int(x.size(0) / x.size(2))
        im = im.resize((imsize, int(imsize * imsize_ratio)), Image.NEAREST)
    else:
        imsize_ratio = int(x.size(2) / x.size(0))
        im = im.resize((int(imsize * imsize_ratio), imsize), Image.NEAREST)
    im.save(path)


# =============================================================================
# generate a grid of images

def make_image_grid(x, ngrid):
    
    x = x.clone().cpu()
    if pow(ngrid,2) < x.size(0):
        grid = make_grid(x[:ngrid*ngrid], nrow=ngrid, padding=0, normalize=True, scale_each=False)
    else:
        grid = torch.FloatTensor(ngrid*ngrid, x.size(1), x.size(2), x.size(3)).fill_(1)
        grid[:x.size(0)].copy_(x)
        grid = make_grid(grid, nrow=ngrid, padding=0, normalize=True, scale_each=False)
    return grid


# =============================================================================

# =============================================================================
# generate a grid of depth maps

def make_image_grid_depth(x, ngrid):
    x = x.clone().cpu()
    if pow(ngrid, 2) < x.size(0):
        #grid = make_grid(x[:ngrid * ngrid], nrow=ngrid, padding=0, normalize=False, scale_each=False)
        grid = make_grid(x, nrow=ngrid, padding=0, normalize=False, scale_each=False)
    else:
        grid = torch.FloatTensor(ngrid * ngrid, x.size(1), x.size(2)).fill_(1)
        grid[:x.size(0)].copy_(x)
        grid = make_grid(grid, nrow=ngrid, padding=0, normalize=False, scale_each=False)
    return grid[0,:,:] #make_grid automatically copies a 1-channel image to 3 channels, so we only want the first channel here


# =============================================================================

# get a grid of images
    
def get_image_grid(x, imsize=512, ngrid=4, color='', size=2):
    
    from PIL import Image
    grid = make_image_grid(x, ngrid)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((imsize,imsize), Image.NEAREST)
    if color is not '': 
        im = add_border(im, color, size)
    im = np.array(im)
    return im


# =============================================================================
# get a grid of depth maps

def get_image_grid_depth(x, imsize=512, ngrid=4, color='', size=2):
    from PIL import Image
    grid = make_image_grid_depth(x, ngrid)
    #ndarr = np.array(grid.add(1).mul(5000).clamp(0, 10000).to(torch.int32).numpy(), dtype='int32')
    ndarr = grid.add(1).mul(5000/255).clamp(0, 255).byte().numpy()
    ndarr = ndarr*(int(255/np.max(ndarr))) # Normalize
    im = Image.fromarray(ndarr)
    im = im.resize((imsize, imsize), Image.NEAREST)
    if color is not '':
        im = add_border(im, color, size)
    im = np.array(im)
    return im#ndarr

# =============================================================================
# save a grid of images
    
def save_image_grid(x, path, imsize=512, ngrid=4, color='', size=2):
    
    from PIL import Image
    grid = make_image_grid(x, ngrid)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((imsize,imsize), Image.NEAREST)
    if color is not '': 
        im = add_border(im, border=size, fill=color)
    im.save(path)


# =============================================================================
# save a grid of depth maps

def save_image_grid_depth(x, path, imsize=512, ngrid=4, color='', size=2):
    #print(x.shape)
    from PIL import Image
    grid = make_image_grid_depth(x, ngrid)
    ndarr = np.array(grid.add(1).mul(5000).clamp(0, 10000).to(torch.int32).numpy(), dtype='int32')
    #print(ndarr.shape)
    im = Image.fromarray(ndarr,"I")
    #im = im.resize((imsize, imsize), Image.NEAREST)
    #if color is not '':
    #    im = add_border(im, border=size, fill=color)
    im.save(path)

# =============================================================================
# add colored border to image
    
def add_border(x, color='', size=2):
    
    from PIL import ImageOps
    if color is not '': 
        x = ImageOps.expand(x, border=size, fill=color)
    return x
        

# =============================================================================
# calculate output dimension of convolution
    
def get_out_dim_conv(dim, k, stride, pad):
    x = ((dim+2*pad-1*(k-1)-1)/stride)+1
    return x


# =============================================================================
# calculate output dimension of transposed convolution
    
def get_out_dim_conv_transpose(dim, k, stride, pad):
    x = (dim-1)*stride-2*pad+k
    return x


# =============================================================================
# count model parameters
    
def count_model_params(model):
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)