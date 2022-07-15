import detectron2.data.transforms as T
from torchvision import transforms

from .transforms.augmentation_impl import GaussianBlur


def build_strong_augmentation(is_train):
    """ Create a list of strong aumentations """

    augmentation = []

    if is_train:
        # Add the color jittering
        augmentation.append(transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], 0.8))

        # Add the grayscale
        augmentation.append(transforms.RandomGrayscale(0.2))

        # Add the Gaussian blur
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 0.2])], 0.5))

        # Add the 3 cutout patterns
        # Convert a PIL image to pytorch tensor because does not support PIL image
        augmentation.append(transforms.Compose([transforms.ToTensor(),
                                                transforms.RandomErasing(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value='random'),
                                                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value='random'),
                                                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value='random'),
                                                transforms.ToPILImage()]))
    
    return transforms.Compose(augmentation)
