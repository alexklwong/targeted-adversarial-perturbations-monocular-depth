'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Safa Cicek <safacicek@ucla.edu>
If this code is useful to you, please consider citing the following paper:
A. Wong, S. Cicek, and S. Soatto. Targeted Adversarial Perturbations for Monocular Depth Prediction.
https://arxiv.org/pdf/2006.08602.pdf
@inproceedings{wong2020targeted,
    title={Targeted Adversarial Perturbations for Monocular Depth Prediction},
    author={Wong, Alex and Safa Cicek and Soatto, Stefano},
    booktitle={Advances in neural information processing systems},
    year={2020}
}
'''
import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator


def read_paths(filepath):
    '''
    Stores a depth map into an image (16 bit PNG)

    Args:
        path : str
            path to file where data will be stored
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')

            # If there was nothing to read
            if path == '':
                break

            path_list.append(path)

    return path_list

def load_image(path, shape=None, normalize=True):
    '''
    Loads an RGB image

    Args:
        path : str
            path to RGB image
        shape : tuple
            tuple of height and width (H, W)
        normalize : bool
            if set, then normalize image between [0, 1]

    Returns:
        numpy : N x H x W x C image
        tuple : original height and width (H, W)
    '''

    # Load image
    image = Image.open(path).convert('RGB')
    o_width, o_height = image.size

    # Resize image
    if shape is not None:
        n_height, n_width = shape
        image = image.resize((n_width, n_height), Image.LANCZOS)

    image = np.asarray(image, np.float32)
    image = np.transpose(image, (2, 0, 1))

    # Normalize
    image = image / 255.0 if normalize else image

    return image, (o_height, o_width)

def load_class_mask(path, shape=None):
    '''
    Loads an RGB image

    Args:
        path : str
            path to RGB image
        shape : tuple
            tuple of height and width (H, W)

    Returns:
        numpy : N x H x W x C image
    '''

    # Load class mask and resize
    class_mask = Image.open(path)

    if shape is not None:
        n_height, n_width = shape
        class_mask = class_mask.resize((n_width, n_height), Image.NEAREST)

    # Reshape to (1, H, W)
    class_mask = np.expand_dims(np.asarray(class_mask, np.float32), 0)

    return np.where(class_mask > 0, 1.0, 0.0).astype(np.float32)

def interpolate_depth(depth_map, validity_map, log_space=False):
    '''
    Interpolate sparse depth with barycentric coordinates

    Args:
        depth_map : numpy
            H x W depth map
        validity_map : numpy
            H x W depth map
        log_space : bool
            if set then produce in log space

    Returns:
        numpy : H x W interpolated depth map
    '''

    assert depth_map.ndim == 2 and validity_map.ndim == 2

    rows, cols = depth_map.shape
    data_row_idx, data_col_idx = np.where(validity_map)
    depth_values = depth_map[data_row_idx, data_col_idx]

    # Perform linear interpolation in log space
    if log_space:
        depth_values = np.log(depth_values)

    interpolator = LinearNDInterpolator(
        # points=Delaunay(np.stack([data_row_idx, data_col_idx], axis=1).astype(np.float32)),
        points=np.stack([data_row_idx, data_col_idx], axis=1),
        values=depth_values,
        fill_value=0 if not log_space else np.log(1e-3))

    query_row_idx, query_col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    query_coord = np.stack([query_row_idx.ravel(), query_col_idx.ravel()], axis=1)
    Z = interpolator(query_coord).reshape([rows, cols])

    if log_space:
        Z = np.exp(Z)
        Z[Z < 1e-1] = 0.0

    return Z
