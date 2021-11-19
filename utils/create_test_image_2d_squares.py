import logging
import os
import sys
import tempfile
from glob import glob


import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    EnsureType,
)
from PIL import Image, ImageDraw

from monai.visualize import plot_2d_or_3d_image
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Tuple
from monai.transforms.utils import rescale_array


def get_rect(x, y, width, height, angle):
    rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    theta = (np.pi / 180.0) * angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect

def get_data():
    """Make an array for the demonstration."""
    X, Y = np.meshgrid(np.linspace(0, np.pi, 128), np.linspace(0, 2, 128))
    z = (np.sin(X) + np.cos(Y)) ** 2 + 0.25
    data = (255 * (z / z.max())).astype(int)
    return data*0


def create_test_image_2d_squares(
    width: int,
    height: int,
    num_objs: int = 12,
    rad_max: int = 30,
    rad_min: int = 5,
    noise_max: float = 0.0,
    num_seg_classes: int = 5,
    channel_dim: Optional[int] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return an image with `num_objs` squares and a 2D mask image. 
    Args:
        width: width of the image. The value should be larger than `2 * rad_max`.
        height: height of the image. The value should be larger than `2 * rad_max`.
        num_objs: number of squares to generate. Defaults to `12`.
        noise_max: if greater than 0 then noise will be added to the image taken from
            the uniform distribution on range `[0,noise_max)`. Defaults to `0`.
        num_seg_classes: number of classes for segmentations. Defaults to `5`.
        channel_dim: if None, create an image without channel dimension, otherwise create
            an image with channel dimension as first dim or last dim. Defaults to `None`.
        random_state: the random generator to use. Defaults to `np.random`.
    """
    image = get_data().astype('float')
    rs: np.random.RandomState = np.random.random.__self__ if random_state is None else random_state  # type: ignore
    img = Image.fromarray(data)
    draw = ImageDraw.Draw(img)
    for _ in range(num_objs):
        x = rs.randint(0, 128)
        y = rs.randint(0, 128)
        width = rs.randint(10, 40)
        height = rs.randint(10, 50)

        rad = rs.randint(rad_min, rad_max)
        angle = rs.randint(0, 180)
        rect = get_rect(x=x, y=y, width=width, height=height, angle=angle)
        draw.polygon([tuple(p) for p in rect], fill=rs.random() * 0.5 + 0.5)
        new_data = np.asarray(img)
  
    labels = np.ceil(new_data).astype(np.int32)
    
    norm = rs.uniform(0, num_seg_classes * noise_max, size=new_data.shape)
    noisyimage: np.ndarray = rescale_array(np.maximum(new_data, norm))  # type: ignore

    if channel_dim is not None:
        if not (isinstance(channel_dim, int) and channel_dim in (-1, 0, 2)):
            raise AssertionError("invalid channel dim.")
        if channel_dim == 0:
            noisyimage = noisyimage[None]
            labels = labels[None]
        else:
            noisyimage = noisyimage[..., None]
            labels = labels[..., None]
    return np.array(noisyimage), np.array(labels)