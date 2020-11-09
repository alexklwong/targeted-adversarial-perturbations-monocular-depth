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
import torch.utils.data
import data_utils


def augment_image_color(images,
                        color_range=None,
                        intensity_range=None,
                        gamma_range=None,
                        normalize=True):
    '''
    Performs color augmentation

    Args:
        images : list[numpy]
            C x H x W or H x W x C images
        color_range : list[float]
            min and max color augmentation range
        intensity_range : list[float]
            min and max intensity augmentation range
        gamma_range : list[float]
            min and max gamma augmentation range
        normalize : bool
            if set, then normalize image between [0, 1]

    Returns:
        list[numpy] : color augmented images
    '''

    if color_range is not None:
        color = np.random.uniform(color_range[0], color_range[1], 3)
        images = [np.reshape(color, [3, 1, 1]) * image for image in images]

    if intensity_range is not None:
        intensity = np.random.uniform(intensity_range[0], intensity_range[1], 1)
        images = [intensity * image for image in images]

    if gamma_range is not None:
        gamma = np.random.uniform(gamma_range[0], gamma_range[1], 1)
        images = [np.power(image, gamma) for image in images]

    if normalize:
        images = [np.clip(image, 0.0, 1.0).astype(np.float32) for image in images]
    else:
        images = [np.clip(image, 0.0, 255.0).astype(np.float32) for image in images]

    return images


class ImagePairCameraDataset(torch.utils.data.Dataset):

    def __init__(self,
                 image0_paths,
                 image1_paths,
                 camera_paths,
                 shape,
                 normalize=True,
                 augment=False,
                 color_range=[0.5, 2.0],
                 intensity_range=[0.8, 1.2],
                 gamma_range=[0.8, 1.2]):

        '''
        Dataset to return image pair and camera (focal length, baseline)

        Args:
            image0_paths : list[str]
                list of paths to left images
            image0_paths : list[str]
                list of paths to right images
            camera_paths : list[str]
                list of paths to focal length and baseline
            shape : tuple[int]
                tuple of height, width
            normalize : bool
                if set, then normalize image between [0, 1]
            color_range : list[float]
                min and max color augmentation range
            intensity_range : list[float]
                min and max intensity augmentation range
            gamma_range : list[float]
                min and max gamma augmentation range

        Returns:
            numpy : C x H x W image
            numpy : C x H x W image
            numpy : scaling factor from focal length baseline
        '''

        self.image0_paths = image0_paths
        self.image1_paths = image1_paths
        self.camera_paths = camera_paths
        self.shape = shape
        self.normalize = normalize
        self.augment = augment
        self.color_range = color_range
        self.intensity_range = intensity_range
        self.gamma_range = gamma_range

    def __getitem__(self, index):
        # Load image
        image0, (height, width) = data_utils.load_image(
            self.image0_paths[index],
            shape=self.shape,
            normalize=self.normalize)
        image1, _  = data_utils.load_image(
            self.image1_paths[index],
            shape=self.shape,
            normalize=self.normalize)

        # Load camera (focal length and baseline)
        camera = np.load(self.camera_paths[index]).astype(np.float32)
        camera = np.prod(camera) / np.asarray(width, np.float32)
        camera = np.reshape(camera, list(camera.shape) + [1, 1, 1])

        # Color augmentation
        if self.augment and np.random.uniform(0.0, 1.0, 1) > 0.50:
            image0, image1 = augment_image_color(
                [image0, image1],
                color_range=self.color_range,
                intensity_range=self.intensity_range,
                gamma_range=self.gamma_range,
                normalize=self.normalize)

        return image0, image1, camera

    def __len__(self):
        return len(self.image0_paths)
