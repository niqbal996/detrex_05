from detrex.config import get_config
from .models.deta_swin import model

# dataloader = get_config("common/data/sugarbeets.py").dataloader

from configs.common.data.sugarbeets_augmented import dataloader
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# 24ep for finetuning
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_24ep
train.init_checkpoint = "/netscratch/naeem/converted_deta_swin_o365_finetune.pth"
# train.init_checkpoint = "/netscratch/naeem/attention_maps_v6/augmentation_experiments/phenobench_def_detr_syn_v6_aug_MedianBlur/model_best_mAP50.pth"
train.output_dir = "/netscratch/naeem/attention_maps_v6/augmentation_experiments/"
# modify learning rate
optimizer.lr = 5e-5
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# max training iterations
train.max_iter = 5000
train.eval_period = 200
train.log_period = 10
train.checkpointer = dict(period=200, max_to_keep=1)

# set training devices
train.device = "cuda"
model.device = train.device
model.num_classes = 2 # crop, weeds

dataloader.train.total_batch_size = 6          # 30 for ResNet50 and 8 for SWIN backbone on 80GB 
dataloader.train.num_workers = 2
dataloader.test.batch_size = 2
dataloader.test.num_workers = 1

# dataloader.train.total_batch_size = 3          # 30 for ResNet50 and 8 for SWIN backbone on 80GB 
# dataloader.train.num_workers = 1
# dataloader.test.batch_size = 1
# dataloader.test.num_workers = 1

from detectron2.data.datasets import register_coco_instances
register_coco_instances("pheno_train", {},
                        "/netscratch/naeem/phenobench/coco_annotations/coco_plants_panoptic_train.json",
                        "/netscratch/naeem/phenobench/train/")
register_coco_instances("syn_pheno_train", {},
                        "/netscratch/naeem/sugarbeet_syn_v6/coco_annotations/instances_train.json",
                        "/netscratch/naeem/sugarbeet_syn_v6/images") # Albumentations output
register_coco_instances("pheno_val", {},
                        "/netscratch/naeem/phenobench/coco_annotations/coco_plants_panoptic_val.json",
                        "/netscratch/naeem/phenobench/val/")