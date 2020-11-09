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
import os, glob
import numpy as np
from scipy import ndimage
from PIL import Image


KITTI_SEMANTICS_ROOT_DIRPATH = os.path.join(
    'data', 'kitti_data_semantics', 'training')
KITTI_SEMANTICS_IMAGE_DIRPATH = os.path.join(
    KITTI_SEMANTICS_ROOT_DIRPATH, 'image_2')
KITTI_SEMANTICS_SEMANTIC_DIRPATH = os.path.join(
    KITTI_SEMANTICS_ROOT_DIRPATH, 'semantic_rgb')
KITTI_SEMANTICS_INSTANCE_DIRPATH = os.path.join(
    KITTI_SEMANTICS_ROOT_DIRPATH, 'instance')

TEST_REFS_DIRPATH = 'testing'

OUTPUT_ROOT_DIRPATH = os.path.join(
    'data', 'kitti_data_semantics_targeted_adversarial', 'training', 'instance')

TEST_INSTANCE_IMAGE_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, 'kitti_instance_test_image.txt')
TEST_INSTANCE_LABEL_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, 'kitti_instance_test_label.txt')

if not os.path.exists(OUTPUT_ROOT_DIRPATH):
    os.makedirs(OUTPUT_ROOT_DIRPATH)

if not os.path.exists(TEST_REFS_DIRPATH):
    os.makedirs(TEST_REFS_DIRPATH)

image_paths = sorted(glob.glob(os.path.join(
    KITTI_SEMANTICS_IMAGE_DIRPATH, '*.png')))
label_paths = sorted(glob.glob(os.path.join(
    KITTI_SEMANTICS_INSTANCE_DIRPATH, '*.png')))

assert len(image_paths) == len(label_paths)

SELECT_INSTANCE_SAMPLES = [
    ['000002_10.png', [(25, 1), (33, 1)], (29, 85)],
    ['000003_10.png', [(25, 1), (33, 2)], (17, 51)],
    ['000016_10.png', [(26, 2), (26, 7), (24, 1), (28, 1)], (41, 61)],
    ['000047_10.png', [(26, 4)], (19, 39)],
    ['000060_10.png', [(27, 1)], (15, 35)],
    ['000086_10.png', [(26, 2)], (23, 39)],
    ['000094_10.png', [(26, 12), (26, 13), (26, 18),
                       (26, 19), (26, 21), (26, 22)], (31, 55)],
    ['000102_10.png', [(26, 11)], (29, 51)],
    ['000118_10.png', [(26, 4), (26, 10), (26, 11), (26, 12),
                       (26, 13), (26, 14), (26, 15)], (15, 35)],
    ['000141_10.png', [(26, 4)], (21, 45)],
    ['000165_10.png', [(26, 5)], (17, 29)],
    ['000169_10.png', [(24, 3), (24, 2)], (33, 95)],
    ['000196_10.png', [(26, 2)], (23, 47)],
]


image_output_paths = []
label_output_paths = []
for image_path, label_path in zip(image_paths, label_paths):
    _, image_filename = os.path.split(image_path)
    _, label_filename = os.path.split(label_path)
    assert image_filename == label_filename

    image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.uint8)
    label = np.asarray(Image.open(label_path))
    instance_label = label % 256
    category_label = label // 256

    shape = image.shape
    output_map = np.zeros([shape[0], shape[1]])

    for filename, labels, dilate_rate in SELECT_INSTANCE_SAMPLES:
        if filename == image_filename:
            for category, instance in labels:
                select = np.logical_and(
                    category_label == category, instance_label == instance)
                output_map = np.logical_or(output_map, select)

            label_output_path = image_path.replace(
                KITTI_SEMANTICS_IMAGE_DIRPATH, OUTPUT_ROOT_DIRPATH)
            image_output_paths.append(image_path)
            label_output_paths.append(label_output_path)

            # Monodepth2 does not respect edge boundaries and oversmooths
            output_map = ndimage.grey_dilation(output_map, size=dilate_rate)
            output_map = (255.0 * output_map).astype(np.uint8)
            Image.fromarray(output_map).save(label_output_path)

print('Storing {} instance images file paths into: {}'.format(
    len(image_output_paths), TEST_INSTANCE_IMAGE_FILEPATH))
with open(TEST_INSTANCE_IMAGE_FILEPATH, "w") as o:
    for idx in range(len(image_output_paths)):
        o.write(image_output_paths[idx] + '\n')

print('Storing {} instance labels file paths into: {}'.format(
    len(label_output_paths), TEST_INSTANCE_LABEL_FILEPATH))
with open(TEST_INSTANCE_LABEL_FILEPATH, "w") as o:
    for idx in range(len(label_output_paths)):
        o.write(label_output_paths[idx] + '\n')
