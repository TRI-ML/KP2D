# Copyright 2020 Toyota Research Institute.  All rights reserved.

import random
from math import pi

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from kp2d.utils.image import image_grid

def filter_dict(dict, keywords):
    """
    Returns only the keywords that are part of a dictionary

    Parameters
    ----------
    dictionary : dict
        Dictionary for filtering
    keywords : list of str
        Keywords that will be filtered

    Returns
    -------
    keywords : list of str
        List containing the keywords that are keys in dictionary
    """
    return [key for key in keywords if key in dict]


def resize_sample(sample, image_shape, image_interpolation=Image.ANTIALIAS):
    """
    Resizes a sample, which contains an input image.

    Parameters
    ----------
    sample : dict
        Dictionary with sample values (output from a dataset's __getitem__ method)
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    # image
    image_transform = transforms.Resize(image_shape, interpolation=image_interpolation)
    sample['image'] = image_transform(sample['image'])
    return sample


def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.

    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to

    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """
    sample['image'] = sample['image'].type(tensor_type)
    sample['image_aug'] = sample['image_aug'].type(tensor_type)
    return sample


def spatial_augment_sample(sample):
    """ Apply spatial augmentation to an image (flipping and random affine transformation)."""
    augment_image = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1))
        
    ])
    sample['image'] = augment_image(sample['image'])

    return sample

def unnormalize_image(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """ Counterpart method of torchvision.transforms.Normalize."""
    for t, m, s in zip(tensor, mean, std):
        t.div_(1 / s).sub_(-m)
    return tensor


def sample_homography(
        shape, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=100, n_angles=100, scaling_amplitude=0.1, perspective_amplitude=0.4,
        patch_ratio=0.8, max_angle=pi/4):
    """ Sample a random homography that includes perspective, scale, translation and rotation operations."""

    width = float(shape[1])
    hw_ratio = float(shape[0]) / float(shape[1])

    pts1 = np.stack([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], axis=0)
    pts2 = pts1.copy() * patch_ratio
    pts2[:,1] *= hw_ratio

    if perspective:

        perspective_amplitude_x = np.random.normal(0., perspective_amplitude/2, (2))
        perspective_amplitude_y = np.random.normal(0., hw_ratio * perspective_amplitude/2, (2))

        perspective_amplitude_x = np.clip(perspective_amplitude_x, -perspective_amplitude/2, perspective_amplitude/2)
        perspective_amplitude_y = np.clip(perspective_amplitude_y, hw_ratio * -perspective_amplitude/2, hw_ratio * perspective_amplitude/2)

        pts2[0,0] -= perspective_amplitude_x[1]
        pts2[0,1] -= perspective_amplitude_y[1]

        pts2[1,0] -= perspective_amplitude_x[0]
        pts2[1,1] += perspective_amplitude_y[1]

        pts2[2,0] += perspective_amplitude_x[1]
        pts2[2,1] -= perspective_amplitude_y[0]

        pts2[3,0] += perspective_amplitude_x[0]
        pts2[3,1] += perspective_amplitude_y[0]

    if scaling:

        random_scales = np.random.normal(1, scaling_amplitude/2, (n_scales))
        random_scales = np.clip(random_scales, 1-scaling_amplitude/2, 1+scaling_amplitude/2)

        scales = np.concatenate([[1.], random_scales], 0)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = np.expand_dims(pts2 - center, axis=0) * np.expand_dims(
                np.expand_dims(scales, 1), 1) + center
        valid = np.arange(n_scales)  # all scales are valid except scale=1
        idx = valid[np.random.randint(valid.shape[0])]
        pts2 = scaled[idx]

    if translation:
        t_min, t_max = np.min(pts2 - [-1., -hw_ratio], axis=0), np.min([1., hw_ratio] - pts2, axis=0)
        pts2 += np.expand_dims(np.stack([np.random.uniform(-t_min[0], t_max[0]),
                                         np.random.uniform(-t_min[1], t_max[1])]),
                               axis=0)

    if rotation:
        angles = np.linspace(-max_angle, max_angle, n_angles)
        angles = np.concatenate([[0.], angles], axis=0) 

        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul(
                np.tile(np.expand_dims(pts2 - center, axis=0), [n_angles+1, 1, 1]),
                rot_mat) + center

        valid = np.where(np.all((rotated >= [-1.,-hw_ratio]) & (rotated < [1.,hw_ratio]),
                                        axis=(1, 2)))[0]

        idx = valid[np.random.randint(valid.shape[0])]
        pts2 = rotated[idx]

    pts2[:,1] /= hw_ratio

    def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]
    def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.transpose(np.stack(
        [[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))

    homography = np.matmul(np.linalg.pinv(a_mat), p_mat).squeeze()
    homography = np.concatenate([homography, [1.]]).reshape(3,3)
    return homography

def warp_homography(sources, homography):
    """Warp features given a homography

    Parameters
    ----------
    sources: torch.tensor (1,H,W,2)
        Keypoint vector.
    homography: torch.Tensor (3,3)
        Homography.

    Returns
    -------
    warped_sources: torch.tensor (1,H,W,2)
        Warped feature vector.
    """
    _, H, W, _ = sources.shape
    warped_sources = sources.clone().squeeze()
    warped_sources = warped_sources.view(-1,2)
    warped_sources = torch.addmm(homography[:,2], warped_sources, homography[:,:2].t())
    warped_sources.mul_(1/warped_sources[:,2].unsqueeze(1))
    warped_sources = warped_sources[:,:2].contiguous().view(1,H,W,2)
    return warped_sources

def add_noise(img, mode="gaussian", percent=0.02):
    """Add image noise

    Parameters
    ----------
    image : np.array
        Input image
    mode: str
        Type of noise, from ['gaussian','salt','pepper','s&p']
    percent: float
        Percentage image points to add noise to.
    Returns
    -------
    image : np.array
        Image plus noise.
    """
    original_dtype = img.dtype
    if mode == "gaussian":
        mean = 0
        var = 0.1
        sigma = var * 0.5

        if img.ndim == 2:
            h, w = img.shape
            gauss = np.random.normal(mean, sigma, (h, w))
        else:
            h, w, c = img.shape
            gauss = np.random.normal(mean, sigma, (h, w, c))

        if img.dtype not in [np.float32, np.float64]:
            gauss = gauss * np.iinfo(img.dtype).max
            img = np.clip(img.astype(np.float) + gauss, 0, np.iinfo(img.dtype).max)
        else:
            img = np.clip(img.astype(np.float) + gauss, 0, 1)

    elif mode == "salt":
        print(img.dtype)
        s_vs_p = 1
        num_salt = np.ceil(percent * img.size * s_vs_p)
        coords = tuple([np.random.randint(0, i - 1, int(num_salt)) for i in img.shape])

        if img.dtype in [np.float32, np.float64]:
            img[coords] = 1
        else:
            img[coords] = np.iinfo(img.dtype).max
            print(img.dtype)
    elif mode == "pepper":
        s_vs_p = 0
        num_pepper = np.ceil(percent * img.size * (1.0 - s_vs_p))
        coords = tuple(
            [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        )
        img[coords] = 0

    elif mode == "s&p":
        s_vs_p = 0.5

        # Salt mode
        num_salt = np.ceil(percent * img.size * s_vs_p)
        coords = tuple([np.random.randint(0, i - 1, int(num_salt)) for i in img.shape])
        if img.dtype in [np.float32, np.float64]:
            img[coords] = 1
        else:
            img[coords] = np.iinfo(img.dtype).max

        # Pepper mode
        num_pepper = np.ceil(percent * img.size * (1.0 - s_vs_p))
        coords = tuple(
            [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        )
        img[coords] = 0
    else:
        raise ValueError("not support mode for {}".format(mode))

    noisy = img.astype(original_dtype)
    return noisy


def non_spatial_augmentation(img_warp_ori, jitter_paramters, color_order=[0,1,2], to_gray=False):
    """ Apply non-spatial augmentation to an image (jittering, color swap, convert to gray scale, Gaussian blur)."""

    brightness, contrast, saturation, hue = jitter_paramters
    color_augmentation = transforms.ColorJitter()
    augment_image = color_augmentation.get_params(brightness=[max(0, 1 - brightness), 1 + brightness],
                                                    contrast=[max(0, 1 - contrast), 1 + contrast],
                                                    saturation=[max(0, 1 - saturation), 1 + saturation],
                                                    hue=[-hue, hue])


    B = img_warp_ori.shape[0]
    img_warp = []
    kernel_sizes = [0,1,3,5]
    for b in range(B):
        img_warp_sub = img_warp_ori[b].cpu()
        img_warp_sub = torchvision.transforms.functional.to_pil_image(img_warp_sub)

        img_warp_sub_np = np.array(img_warp_sub) 
        img_warp_sub_np = img_warp_sub_np[:,:,color_order]
        
        if np.random.rand() > 0.5:
            img_warp_sub_np = add_noise(img_warp_sub_np)

        rand_index = np.random.randint(4)
        kernel_size = kernel_sizes[rand_index]
        if kernel_size >0:
            img_warp_sub_np = cv2.GaussianBlur(img_warp_sub_np, (kernel_size, kernel_size), sigmaX=0)
        
        if to_gray:
            img_warp_sub_np = cv2.cvtColor(img_warp_sub_np, cv2.COLOR_RGB2GRAY)
            img_warp_sub_np = cv2.cvtColor(img_warp_sub_np, cv2.COLOR_GRAY2RGB)

        img_warp_sub = Image.fromarray(img_warp_sub_np)
        img_warp_sub = color_augmentation(img_warp_sub)

        img_warp_sub = torchvision.transforms.functional.to_tensor(img_warp_sub).to(img_warp_ori.device)

        img_warp.append(img_warp_sub)

    img_warp = torch.stack(img_warp, dim=0)
    return img_warp

def ha_augment_sample(data, jitter_paramters=[0.5, 0.5, 0.2, 0.05], patch_ratio=0.7, scaling_amplitude=0.2, max_angle=pi/4):
    """Apply Homography Adaptation image augmentation."""
    target_img = data['image'].unsqueeze(0)
    _, _, H, W = target_img.shape
    device = target_img.device

    # Generate homography (warps source to target)
    homography = sample_homography([H, W],
        patch_ratio=patch_ratio, 
        scaling_amplitude=scaling_amplitude, 
        max_angle=max_angle)
    homography = torch.from_numpy(homography).float().to(device)

    source_grid = image_grid(1, H, W,
                    dtype=target_img.dtype,
                    device=device,
                    ones=False, normalized=True).clone().permute(0, 2, 3, 1)

    source_warped = warp_homography(source_grid, homography)
    source_img = torch.nn.functional.grid_sample(target_img, source_warped, align_corners=True)

    color_order = [0,1,2]
    if np.random.rand() > 0.5:
        random.shuffle(color_order)

    to_gray = False
    if np.random.rand() > 0.5:
        to_gray = True

    target_img = non_spatial_augmentation(target_img, jitter_paramters=jitter_paramters, color_order=color_order, to_gray=to_gray)
    source_img = non_spatial_augmentation(source_img, jitter_paramters=jitter_paramters, color_order=color_order, to_gray=to_gray)

    data['image'] = target_img.squeeze()
    data['image_aug'] = source_img.squeeze()
    data['homography'] = homography
    return data
def ha_augment_real_sonar_sample(data, jitter_paramters=[0.5, 0.5, 0.2, 0.05], patch_ratio=0.7, scaling_amplitude=0.2, max_angle=pi/4):
    """Apply Homography Adaptation image augmentation."""
    target_img = data['image'].unsqueeze(0)
    _, _, H, W = target_img.shape
    device = target_img.device

    # Generate homography (warps source to target)
    homography = sample_homography([H, W],
        patch_ratio=patch_ratio,
        scaling_amplitude=scaling_amplitude,
        max_angle=0)
    homography = torch.from_numpy(homography).float().to(device)

    source_grid = image_grid(1, H, W,
                    dtype=target_img.dtype,
                    device=device,
                    ones=False, normalized=True).clone().permute(0, 2, 3, 1)

    source_warped = warp_homography(source_grid, homography)
    source_img = torch.nn.functional.grid_sample(target_img, source_warped, align_corners=True)

    #TODO: add cone representation do rotation and translation not scaling (implement shear, wavy and trapez) <- probably not working

    #TODO: add some noise
    color_order = [0,1,2]
    to_gray = False

    target_img = non_spatial_augmentation(target_img, jitter_paramters=jitter_paramters, color_order=color_order, to_gray=to_gray)
    source_img = non_spatial_augmentation(source_img, jitter_paramters=jitter_paramters, color_order=color_order, to_gray=to_gray)

    data['image'] = target_img.squeeze()
    data['image_aug'] = source_img.squeeze()
    data['homography'] = homography
    return data

def ha_augment_sonar_sim_sample(data, jitter_paramters=[0.5, 0.5, 0.2, 0.05], patch_ratio=0.7, scaling_amplitude=0.2, max_angle=pi/4):
    """Apply Homography Adaptation image augmentation."""
    target_img = data['image'].unsqueeze(0)
    _, _, H, W = target_img.shape
    device = target_img.device

    # Generate homography (warps source to target)
    homography = sample_homography([H, W],
        patch_ratio=patch_ratio,
        scaling_amplitude=scaling_amplitude,
        max_angle=max_angle)
    homography = torch.from_numpy(homography).float().to(device)

    source_grid = image_grid(1, H, W,
                    dtype=target_img.dtype,
                    device=device,
                    ones=False, normalized=True).clone().permute(0, 2, 3, 1)

    source_warped = warp_homography(source_grid, homography)
    source_img = torch.nn.functional.grid_sample(target_img, source_warped, align_corners=True)

    #TODO: implement shear, wavy and trapez

    #TODO: add sparkle noise, artifacts, row wise noise, normalization row wise,

    #TODO: Blur

    color_order = [0,1,2]
    to_gray = False

    target_img = non_spatial_augmentation(target_img, jitter_paramters=jitter_paramters, color_order=color_order, to_gray=to_gray)
    source_img = non_spatial_augmentation(source_img, jitter_paramters=jitter_paramters, color_order=color_order, to_gray=to_gray)

    data['image'] = target_img.squeeze()
    data['image_aug'] = source_img.squeeze()
    data['homography'] = homography
    return data

