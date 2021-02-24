# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch


@torch.jit.script
def meshgrid(B: int, H: int, W: int, normalized: bool = False):
    """Create mesh-grid given batch size, height and width dimensions.

    Parameters
    ----------
    B: int
        Batch size
    H: int
        Grid Height
    W: int
        Batch size
    dtype: torch.dtype
        Tensor dtype
    device: str
        Tensor device
    normalized: bool
        Normalized image coordinates or integer-grid.

    Returns
    -------
    xs: torch.Tensor
        Batched mesh-grid x-coordinates (BHW).
    ys: torch.Tensor
        Batched mesh-grid y-coordinates (BHW).
    """
    if normalized:
        xs = torch.arange(W).float() / (W - 1) * 2 - 1
        ys = torch.arange(H).float() / (H - 1) * 2 - 1
    else:
        xs = torch.arange(W).float()
        ys = torch.arange(H).float()
    ys, xs = torch.meshgrid([ys, xs])
    return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1])


@torch.jit.script
def image_grid(B: int, H: int, W: int, ones: bool = True, normalized: bool = False):
    """Create an image mesh grid with shape B3HW given image shape BHW

    Parameters
    ----------
    B: int
        Batch size
    H: int
        Grid Height
    W: int
        Batch size
    dtype: str
        Tensor dtype
    device: str
        Tensor device
    ones : bool
        Use (x, y, 1) coordinates
    normalized: bool
        Normalized image coordinates or integer-grid.

    Returns
    -------
    grid: torch.Tensor
        Mesh-grid for the corresponding image shape (B3HW)
    """
    xs, ys = meshgrid(B, H, W, normalized=normalized)
    coords = [xs, ys]
    if ones:
        coords.append(torch.ones_like(xs))  # BHW
    grid = torch.stack(coords, dim=1)  # B3HW
    return grid


def to_gray_normalized(images):
    """Performs image normalization and converts images to grayscale (preserving dimensions)
    
    Parameters
    ----------
    images: torch.Tensor
        Input images.

    Returns
    -------
    normalized_images: torch.Tensor
        Normalized grayscale images.
    """
    assert len(images.shape) == 4
    images -= 0.5
    images *= 0.225
    normalized_images = images.mean(1).unsqueeze(1)
    return normalized_images


def to_color_normalized(images):
    """Performs image normalization and converts images to grayscale (preserving dimensions)
    
    Parameters
    ----------
    images: torch.Tensor
        Input images.

    Returns
    -------
    normalized_images: torch.Tensor
        Normalized grayscale images.
    """
    assert len(images.shape) == 4
    images -= 0.5
    images *= 0.225
    return images
