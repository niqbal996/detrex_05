from detrex.config import get_config
from .models.deformable_detr_r50 import model

dataloader = get_config("common/data/sugarbeets_real.py").dataloader
# lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
from .scheduler.coco_scheduler import lr_multiplier_12ep_synthetic_2e_4 as lr_multiplier
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "/netscratch/naeem/deformable_detr_r50.pth"
train.output_dir = "/netscratch/naeem/attention_maps_v6/phenobench_deformable_detr_real_baseline"

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device
model.num_classes = 2 # crop, weeds

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# max training iterations
train.max_iter = 5000
train.eval_period = 200
train.log_period = 10
train.checkpointer = dict(period=200, max_to_keep=1)

# modify dataloader config
dataloader.train.total_batch_size = 30          # 16 on 40GB and 30 on 80 GB
dataloader.train.num_workers = 20
dataloader.test.batch_size = 30
dataloader.test.num_workers = 20

from detectron2.data.datasets import register_coco_instances
register_coco_instances("pheno_train", {},
                        "/netscratch/naeem/phenobench/coco_annotations/coco_plants_panoptic_train.json",
                        "/netscratch/naeem/phenobench/train/")
register_coco_instances("pheno_val", {},
                        "/netscratch/naeem/phenobench/coco_annotations/coco_plants_panoptic_val.json",
                        "/netscratch/naeem/phenobench/val/")
