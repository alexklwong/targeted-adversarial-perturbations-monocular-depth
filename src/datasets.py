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


class TargetedAdversarialMonocularDataset(torch.utils.data.Dataset):

  def __init__(self,
               image0_paths,
               image1_paths=None,
               class_mask_paths=None,
               shape=None,
               normalize=True):

    self.image0_paths = image0_paths

    placeholder_paths = [None] * len(image0_paths)

    if image1_paths is not None:
        self.image1_paths = image1_paths
    else:
        self.image1_paths = placeholder_paths

    if class_mask_paths is not None:
        self.class_mask_paths = class_mask_paths
    else:
        self.class_mask_paths = placeholder_paths

    self.shape = shape
    self.normalize = normalize

  def __getitem__(self, index):

    # Load images
    image0, (height, width) = data_utils.load_image(
        self.image0_paths[index],
        shape=self.shape,
        normalize=self.normalize)

    if self.image1_paths[index] is not None:
        image1, _ = data_utils.load_image(
            self.image1_paths[index],
            shape=self.shape,
            normalize=self.normalize)
    else:
        image1 = np.zeros_like(image0)

    # Default KITTI baseline, focal length
    scale_factor = \
        np.asarray(0.30725988566827694, np.float32) / np.asarray(width, np.float32)

    scale_factor = np.reshape(scale_factor , list(scale_factor .shape) + [1, 1, 1])

    # Load class map
    if self.class_mask_paths[index] is not None:
        class_mask = data_utils.load_class_mask(
            self.class_mask_paths[index],
            shape=self.shape)
    else:
      class_mask = np.ones([1] + list(self.shape), dtype=np.float32)

    return image0, image1, scale_factor, class_mask

  def __len__(self):
    return len(self.image0_paths)
