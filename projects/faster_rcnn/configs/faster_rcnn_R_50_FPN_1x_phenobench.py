from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.mask_rcnn_fpn import model
from ..common.train import train
from detectron2.data.datasets import register_coco_instances
import datetime
from omegaconf import OmegaConf
import os 
import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="syn_pheno_train"),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(640, 672, 704, 736, 768, 800, 1024),
                sample_style="choice",
                max_size=1333,
            ),
            L(T.RandomFlip)(horizontal=True),
        ],
        image_format="BGR",
        use_instance_mask=True,
    ),
    total_batch_size=4,
    num_workers=8,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="pheno_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=1024, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=18,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)

from detectron2.data.datasets import register_coco_instances
root_dir = '/mnt/e/datasets/'
register_coco_instances("pheno_train", {},
                        os.path.join(root_dir, "coco_annotations/coco_plants_panoptic_train.json"),
                        os.path.join(root_dir, "phenobench/train/")
                        )
register_coco_instances("syn_pheno_train", {},
                        os.path.join(root_dir, "coco_annotations/instances_train.json"),
                        os.path.join(root_dir, "images/")
                )
register_coco_instances("pheno_val", {},
                        os.path.join(root_dir, "coco_annotations/coco_plants_panoptic_val.json"),
                        os.path.join(root_dir, "phenobench/val/")
                        )

dataloader.train.mapper.use_instance_mask = False
dataloader.train.total_batch_size = 1          # 16 on 40GB and 30 on 80 GB
dataloader.train.num_workers = 1
dataloader.test.batch_size = 1
dataloader.test.num_workers = 1

optimizer.lr = 0.01

model.backbone.bottom_up.freeze_at = 0
model.pixel_mean = [136.25, 137.81, 135.14]
model.roi_heads.num_classes = 2
del model.roi_heads.mask_in_features
del model.roi_heads.mask_pooler
del model.roi_heads.mask_head

# max training iterations
train.max_iter = 10000
train.eval_period = 100
train.log_period = 20
train.best=200
train.val_loss_frequency=100
train.checkpointer = dict(period=200, max_to_keep=1)

train.output_dir = '/home/niqbal/git/aa_transformers/detrex/output/frcnn_explainable/'
# train.init_checkpoint = "/netscratch/naeem/model_final_a54504.pkl"
train.init_checkpoint = "/mnt/e/models/faster_rcnn/phenobench/model_final.pth"