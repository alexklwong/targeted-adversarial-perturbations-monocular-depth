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

OUTPUT_ROOT_DIRPATH = os.path.join('data', 'kitti_data_semantics_targeted_adversarial', 'training')

# Flat category
ROAD_LABEL_RGB = (128, 64, 128)
SIDEWALK_LABEL_RGB = (244, 35, 232)
PARKING_LABEL_RGB = (250, 170, 160)
RAILTRACK_LABEL_RGB = (230, 150, 140)
FLAT_LABEL_RGB = [
    ROAD_LABEL_RGB, SIDEWALK_LABEL_RGB, PARKING_LABEL_RGB, RAILTRACK_LABEL_RGB
]
# Construction category
BUILDING_LABEL_RGB = ( 70, 70, 70)
WALL_LABEL_RGB = (102, 102, 156)
FENCE_LABEL_RGB = (190, 153, 153)
GUARDRAIL_LABEL_RGB = (180, 165, 180)
BRIDGE_LABEL_RGB = (150, 100, 100)
TUNNEL_LABEL_RGB = (150, 120, 90)
CONSTRUCTION_LABEL_RGB = [
    BUILDING_LABEL_RGB, WALL_LABEL_RGB, FENCE_LABEL_RGB,
    GUARDRAIL_LABEL_RGB, BRIDGE_LABEL_RGB, TUNNEL_LABEL_RGB
]
# Traffic category
POLE_LABEL_RGB = (153, 153, 153)
TRAFFIC_LIGHT_LABEL_RGB = (250, 170, 30)
TRAFFIC_SIGN_LABEL_RGB = (220, 220, 0)
TRAFFIC_LABEL_RGB = [
    POLE_LABEL_RGB, TRAFFIC_LIGHT_LABEL_RGB, TRAFFIC_SIGN_LABEL_RGB
]
# Nature category
VEGETATION_LABEL_RGB = (107, 142, 35)
TERRAIN_LABEL_RGB = (152, 251, 152)
NATURE_LABEL_RGB = [
    VEGETATION_LABEL_RGB, TERRAIN_LABEL_RGB
]
# Sky category
SKY_LABEL_RGB = [
    (70, 130, 180)
]
# Human category
PERSON_LABEL_RGB = (220, 20, 60)
RIDER_LABEL_RGB = (255, 0, 0)
MOTORCYCLE_LABEL_RGB = (0, 0, 230)
BICYCLE_LABEL_RGB = (119, 11, 32)
HUMAN_LABEL_RGB = [
    PERSON_LABEL_RGB, RIDER_LABEL_RGB, MOTORCYCLE_LABEL_RGB, BICYCLE_LABEL_RGB
]
# Vehicle category
CAR_LABEL_RGB = (0, 0, 142)
TRUCK_LABEL_RGB = (0, 0, 70)
BUS_LABEL_RGB = (0, 60, 100)
CARAVAN_LABEL_RGB = (0, 0, 90)
TRAILER_LABEL_RGB = (0, 0, 110)
TRAIN_LABEL_RGB = (0, 80, 100)
LICENSE_LABEL_RGB = (0, 0, 142)
VEHICLE_LABEL_RGB = [
    CAR_LABEL_RGB, TRUCK_LABEL_RGB, BUS_LABEL_RGB,
    CARAVAN_LABEL_RGB, TRAILER_LABEL_RGB, TRAIN_LABEL_RGB,
    LICENSE_LABEL_RGB
]


image_paths = sorted(glob.glob(os.path.join(KITTI_SEMANTICS_IMAGE_DIRPATH, '*.png')))
label_paths = sorted(glob.glob(os.path.join(KITTI_SEMANTICS_SEMANTIC_DIRPATH, '*.png')))

assert(len(image_paths) == len(label_paths))

TEST_SEMANTIC_IMAGE_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_image.txt')

TEST_SEMANTIC_CONSTRUCTION_IMAGE_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_construction_image.txt')
TEST_SEMANTIC_CONSTRUCTION_LABEL_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_construction_label.txt')

TEST_SEMANTIC_FLAT_IMAGE_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_flat_image.txt')
TEST_SEMANTIC_FLAT_LABEL_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_flat_label.txt')

TEST_SEMANTIC_HUMAN_IMAGE_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_human_image.txt')
TEST_SEMANTIC_HUMAN_LABEL_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_human_label.txt')

TEST_SEMANTIC_NATURE_IMAGE_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_nature_image.txt')
TEST_SEMANTIC_NATURE_LABEL_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_nature_label.txt')

TEST_SEMANTIC_SKY_IMAGE_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_sky_image.txt')
TEST_SEMANTIC_SKY_LABEL_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_sky_label.txt')

TEST_SEMANTIC_TRAFFIC_IMAGE_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_traffic_image.txt')
TEST_SEMANTIC_TRAFFIC_LABEL_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_traffic_label.txt')

TEST_SEMANTIC_VEHICLE_IMAGE_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_vehicle_image.txt')
TEST_SEMANTIC_VEHICLE_LABEL_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_vehicle_label.txt')

TEST_SEMANTIC_NONFLAT_IMAGE_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_nonflat_image.txt')
TEST_SEMANTIC_NONFLAT_LABEL_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_semantic_test_nonflat_label.txt')


data_inputs = [
    [image_paths, label_paths,
      'construction', CONSTRUCTION_LABEL_RGB,
      TEST_SEMANTIC_CONSTRUCTION_IMAGE_FILEPATH, TEST_SEMANTIC_CONSTRUCTION_LABEL_FILEPATH],
    [image_paths, label_paths,
      'flat', FLAT_LABEL_RGB,
      TEST_SEMANTIC_FLAT_IMAGE_FILEPATH, TEST_SEMANTIC_FLAT_LABEL_FILEPATH],
    [image_paths, label_paths,
      'human', HUMAN_LABEL_RGB,
      TEST_SEMANTIC_HUMAN_IMAGE_FILEPATH, TEST_SEMANTIC_HUMAN_LABEL_FILEPATH],
    [image_paths, label_paths,
      'nature', NATURE_LABEL_RGB,
      TEST_SEMANTIC_NATURE_IMAGE_FILEPATH, TEST_SEMANTIC_NATURE_LABEL_FILEPATH],
    [image_paths, label_paths,
      'sky', SKY_LABEL_RGB,
      TEST_SEMANTIC_SKY_IMAGE_FILEPATH, TEST_SEMANTIC_SKY_LABEL_FILEPATH],
    [image_paths, label_paths,
      'traffic', TRAFFIC_LABEL_RGB,
      TEST_SEMANTIC_TRAFFIC_IMAGE_FILEPATH, TEST_SEMANTIC_TRAFFIC_LABEL_FILEPATH],
    [image_paths, label_paths,
      'vehicle', VEHICLE_LABEL_RGB,
      TEST_SEMANTIC_VEHICLE_IMAGE_FILEPATH, TEST_SEMANTIC_VEHICLE_LABEL_FILEPATH],
    [image_paths, label_paths,
      'nonflat', FLAT_LABEL_RGB,
      TEST_SEMANTIC_NONFLAT_IMAGE_FILEPATH, TEST_SEMANTIC_NONFLAT_LABEL_FILEPATH]
]

for data in data_inputs:
    image_input_paths, label_input_paths, \
        category, category_label_rgb, \
        image_output_filepath, label_output_filepath = data

    image_output_paths = []
    label_output_paths = []
    for idx in range(len(image_input_paths)):

        image = np.asarray(Image.open(image_input_paths[idx]).convert('RGB'), np.uint8)
        label = np.asarray(Image.open(label_input_paths[idx]).convert('RGB'), np.uint8)
        shape = image.shape
        category_map = np.zeros([shape[0], shape[1]])

        for label_rgb in category_label_rgb:
            category_map = np.logical_or(category_map, np.all(label == label_rgb, axis=-1))

        if category == 'nonflat':
            # Invert flat category map
            category_map = 1.0 - category_map.astype(np.float32)

        # If this category exists then store
        if np.sum(category_map) > 0:
            image_output_path = image_input_paths[idx]
            label_output_path = label_input_paths[idx] \
                .replace(KITTI_SEMANTICS_ROOT_DIRPATH, os.path.join(OUTPUT_ROOT_DIRPATH, category))

            image_output_paths.append(image_output_path)
            label_output_paths.append(label_output_path)

            label_output_dirpath = os.path.dirname(label_output_path)

            if not os.path.exists(label_output_dirpath):
                os.makedirs(label_output_dirpath)

            category_map = (255.0 * category_map.astype(np.float32)).astype(np.uint8)
            Image.fromarray(category_map).save(label_output_path)

    print('Storing {} {} images file paths into: {}'.format(
        len(image_output_paths), category, image_output_filepath))
    with open(image_output_filepath, "w") as o:
        for idx in range(len(image_output_paths)):
            o.write(image_output_paths[idx]+'\n')

    print('Storing {} {} class map file paths into: {}'.format(
        len(label_output_paths), category, label_output_filepath))
    with open(label_output_filepath, "w") as o:
        for idx in range(len(label_output_paths)):
            o.write(label_output_paths[idx]+'\n')
