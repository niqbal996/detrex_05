from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
import albumentations as A
from detectron2.data import transforms as T
from detectron2.data.transforms import AlbumentationsWrapper
from detectron2.evaluation import COCOEvaluator
import torchvision
dataloader = OmegaConf.create()
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="syn_pheno_train"),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            AlbumentationsWrapper(A.MotionBlur(blur_limit=13, p=1.0)),
            AlbumentationsWrapper(A.Blur(blur_limit=13, p=1.0)),
            AlbumentationsWrapper(A.MedianBlur(blur_limit=13, p=1.0)),
            AlbumentationsWrapper(A.ToGray(p=1.0)),
            AlbumentationsWrapper(A.CLAHE(p=1.0)),
            AlbumentationsWrapper(A.RandomBrightnessContrast(p=1.0)),
            AlbumentationsWrapper(A.ImageCompression(quality_lower=60, quality_upper=70, p=1.0))
        ],
        image_format="BGR",
        use_instance_mask=True,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="pheno_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    max_dets_per_image=500,
)
