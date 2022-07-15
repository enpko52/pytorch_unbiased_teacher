import copy

import torch
from PIL import Image
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils
from detectron2.structures import Boxes
from detectron2.data import DatasetMapper


from .detection_utils import build_strong_augmentation


class DatasetMapperForUBT(DatasetMapper):
    """ The new dataset mapper for training the unbiased teacher model """

    def __init__(self, cfg, is_train=True):
        
        self.is_train = is_train

        # Augmentation
        self.weak_augmentation = T.AugmentationList(utils.build_augmentation(cfg, self.is_train))
        self.strong_augmentation = build_strong_augmentation(self.is_train)


    def __call__(self, dataset_dict):

        # Read an image from dataset dict
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict['file_name'])
        utils.check_image_size(dataset_dict, image)

        # Apply the weak augmentation to the image
        aug_input = T.AugInput(image)
        transforms = self.weak_augmentation(aug_input)
        aug_image = aug_input.image

        dataset_dict['image'] = Image.fromarray(aug_image)
        self._transform_annotations(dataset_dict, transforms, aug_image.shape[:2])
        
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict


        # Apply the strong augmentation to the augmented image
        strong_dataset_dict = copy.deepcopy(dataset_dict)

        aug_image_pil = Image.fromarray(aug_image.astype("uint8"), "RGB")
        strong_image = self.strong_augmentation(aug_image_pil)
        strong_dataset_dict['image'] = strong_image

        assert dataset_dict["image"].size[0] == strong_dataset_dict["image"].size[0]
        assert dataset_dict["image"].size[1] == strong_dataset_dict["image"].size[1]
        return (dataset_dict, strong_dataset_dict)
    
    
    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        """ Update the annotations for the transforms """

        # Create the instances from the dataset dict's annotaitons
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            # for obj in dataset_dict.pop('annotations')
            for obj in dataset_dict['annotations']
        ]
        instances = utils.annotations_to_instances(annos, image_shape)

        # Save the instances
        dataset_dict["instances"] = utils.filter_empty_instances(instances, by_mask=False)
