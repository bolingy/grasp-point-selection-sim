
import numpy as np
import torch
from skimage.transform import resize

def random_crop(depth, seg, out_h, out_w):
    """
    args:
    depth: shape (B, H, W) for depth image
    seg: shape (B, H, W) for segmentation image
    out_h: output height (e.g., 84)
    out_w: output width (e.g., 84)
    """
    n, h, w = depth.shape
    crop_max_h = h - out_h + 1
    crop_max_w = w - out_w + 1
    h1 = np.random.randint(0, crop_max_h, n)
    w1 = np.random.randint(0, crop_max_w, n)
    cropped_depth = np.empty((n, out_h, out_w), dtype=depth.dtype)
    cropped_seg = np.empty((n, out_h, out_w), dtype=seg.dtype)
    
    for i, (depth_img, seg_img, h11, w11) in enumerate(zip(depth, seg, h1, w1)):
        cropped_depth[i] = depth_img[h11:h11 + out_h, w11:w11 + out_w]
        cropped_seg[i] = seg_img[h11:h11 + out_h, w11:w11 + out_w]
    
    # resize to original width and height
    cropped_depth = resize(cropped_depth, (n, h, w), order=0, preserve_range=True)
    cropped_seg = resize(cropped_seg, (n, h, w), order=0, preserve_range=True)
    return cropped_depth, cropped_seg


def random_cutout(depth, seg, min_cut, max_cut):
    """
    args:
    depth: shape (B, H, W) for depth image
    seg: shape (B, H, W) for segmentation image
    min_cut: minimum cut size
    max_cut: maximum cut size
    """
    n, h, w = depth.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)
    
    cutouts_depth = np.empty((n, h, w), dtype=depth.dtype)
    cutouts_seg = np.empty((n, h, w), dtype=seg.dtype)
    
    for i, (depth_img, seg_img, w11, h11) in enumerate(zip(depth, seg, w1, h1)):
        cut_img_depth = depth_img.copy()
        cut_img_depth[h11:h11 + h11, w11:w11 + w11] = 0
        
        cut_img_seg = seg_img.copy()
        cut_img_seg[h11:h11 + h11, w11:w11 + w11] = 0
        
        cutouts_depth[i] = cut_img_depth
        cutouts_seg[i] = cut_img_seg
    
    return cutouts_depth, cutouts_seg

# random flip
import torch
import numpy as np

def random_flip(depth, seg, p=0.5):
    """
    args:
    depth: shape (B, H, W) for depth image
    seg: shape (B, H, W) for segmentation image
    device: torch device
    p: probability of flipping (default is 0.5)
    """
    bs, h, w = depth.shape
    
    # Flip both depth and seg images horizontally
    flipped_depth = np.flip(depth, [2])
    flipped_seg = np.flip(seg, [2])

    # Generate random numbers to decide whether to apply the flip
    rnd = np.random.uniform(0., 1., size=(depth.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)

    # Apply the flip based on the generated mask
    out_depth = np.where(mask.view(-1, 1, 1).expand(bs, h, w), flipped_depth, depth)
    out_seg = np.where(mask.view(-1, 1, 1).expand(bs, h, w), flipped_seg, seg)

    return out_depth, out_seg

