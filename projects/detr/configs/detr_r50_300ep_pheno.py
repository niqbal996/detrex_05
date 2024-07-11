from detrex.config import get_config
from .models.detr_r50 import model

dataloader = get_config("common/data/sugarbeets.py").dataloader
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep_synthetic
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
# train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
# train.init_checkpoint = "/netscratch/naeem/converted_detr_r50_500ep.pth"
train.init_checkpoint = "/netscratch/naeem/phenobench_DETR_syn_v6_augmented/model_final.pth"
train.output_dir = "/netscratch/naeem/phenobench_DETR_syn_v6_augmented"
train.max_iter = 3000
train.eval_period = 200
train.log_period = 10
train.checkpointer = dict(period=200, max_to_keep=1)

# modify optimizer config
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# set training devices
train.device = "cuda"
model.device = train.device
model.num_classes = 2 # crop, weeds

# modify dataloader config
dataloader.train.total_batch_size = 1          # 16 on 40GB and 30 on 80 GB
dataloader.train.num_workers = 1
dataloader.test.batch_size = 1
dataloader.test.num_workers = 1

from detectron2.data.datasets import register_coco_instances
register_coco_instances("pheno_train", {},
                        "/netscratch/naeem/phenobench/coco_annotations/coco_plants_panoptic_train.json",
                        "/netscratch/naeem/phenobench/train/")
register_coco_instances("syn_pheno_train", {},
                        "/netscratch/naeem/sugarbeet_syn_v6/coco_annotations/instances_train.json",
                        "/netscratch/naeem/sugarbeet_syn_v6/images/") # baseline
register_coco_instances("pheno_val", {},
                        "/netscratch/naeem/phenobench/coco_annotations/coco_plants_panoptic_val.json",
                        "/netscratch/naeem/phenobench/val/")
