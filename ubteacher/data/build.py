import operator

from torch.utils.data import DataLoader
from detectron2 import data
from detectron2.data.samplers import *
from detectron2.utils.comm import get_world_size
from detectron2.data.build import worker_init_reset_seed

from .common import AspectRatioGroupedSemiSupDataset


def build_detection_semisup_train_loader(cfg, mapper=None):
    """ Build a dataloader for semi-supervised learning """

    # Get the dataset dicts
    labeled_dicts = data.get_detection_dataset_dicts(
        names=cfg.DATASETS.LABELED_TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None
    )
    unlabeled_dicts = data.get_detection_dataset_dicts(
        names=cfg.DATASETS.UNLABELED_TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None
    )

    # Wrap a list to a torch Dataset
    labeled_dataset = data.DatasetFromList(lst=labeled_dicts, copy=False)
    unlabeled_dataset = data.DatasetFromList(lst=unlabeled_dicts, copy=False)

    # Map a function over the elements in a dataset
    if mapper is None:
        mapper = data.DatasetMapper(cfg=cfg, is_train=True)
    
    labeled_dataset = data.MapDataset(labeled_dataset, mapper)
    unlabeled_dataset = data.MapDataset(unlabeled_dataset, mapper)

    # Create the dataset sampler
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN

    labeled_sampler = eval(sampler_name)(size=len(labeled_dataset))
    unlabeled_sampler = eval(sampler_name)(size=len(unlabeled_dataset))

    # Create the dataloader and return it
    return build_semisup_batch_data_loader(
        dataset=(labeled_dataset, unlabeled_dataset),
        sampler=(labeled_sampler, unlabeled_sampler),
        total_batch_size=(cfg.SOLVER.IMG_PER_BATCH_LABEL, cfg.SOLVER.IMG_PER_BATCH_UNLABEL),
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS
    )


def build_semisup_batch_data_loader(dataset, 
                                    sampler, 
                                    total_batch_size, 
                                    aspect_ratio_grouping=True, 
                                    num_workers=0):
    """ Build a batched dataloader for semi-supervised learning """
    
    world_size = get_world_size()
    assert (
        total_batch_size[0] > 0 and total_batch_size[0] % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size[0], world_size
    )

    assert (
        total_batch_size[1] > 0 and total_batch_size[1] % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size[1], world_size
    )

    # Get the batch sizes
    labeled_batch_size = int(total_batch_size[0] / world_size)
    unlabeled_batch_size = int(total_batch_size[1] / world_size)

    # Get the datasets and the samplers
    labeled_dataset, unlabeled_dataset = dataset
    labeled_sampler, unlabeled_sampler = sampler

    if aspect_ratio_grouping:
        labeled_loader = DataLoader(dataset=labeled_dataset,
                                    batch_size=labeled_batch_size,
                                    sampler=labeled_sampler,
                                    num_workers=num_workers,
                                    collate_fn=operator.itemgetter(0),
                                    worker_init_fn=worker_init_reset_seed)
        unlabeled_loader = DataLoader(dataset=unlabeled_dataset,
                                      batch_size=unlabeled_batch_size,
                                      sampler=unlabeled_sampler,
                                      num_workers=num_workers,
                                      collate_fn=operator.itemgetter(0),
                                      worker_init_fn=worker_init_reset_seed)
        return AspectRatioGroupedSemiSupDataset(
            datasets=(labeled_loader, unlabeled_loader),
            batch_sizes=(labeled_batch_size, unlabeled_batch_size)
        )
    else:
        raise NotImplementedError("ASPECT_RATIO_GROUPING = False is not supported yet")
