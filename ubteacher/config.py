def add_ubteacher_config(cfg):
    """ Add config for unbiased teacher """

    _C = cfg

    # Datasets
    _C.DATASETS.LABELED_TRAIN = ("coco_2017_train",)
    _C.DATASETS.UNLABELED_TRAIN = ("coco_2017_train",) 

    # Solver
    _C.SOLVER.IMG_PER_BATCH_LABEL = 1
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 1

    # Model
    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"