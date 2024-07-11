from detrex.config import get_config
from .models.deta_r50 import model
# from .scheduler.coco_scheduler import lr_multiplier_12ep_10drop as lr_multiplier
from .scheduler.coco_scheduler import lr_multiplier_12ep_pheno as lr_multiplier_real
from .scheduler.coco_scheduler import lr_multiplier_12ep_synthetic as lr_multiplier
# using the default optimizer and dataloader
dataloader = get_config("common/data/sugarbeets.py").dataloader
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
# train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.init_checkpoint = "/netscratch/naeem/phenobench_deta_syn_v4_baseline_CLR/model_best_mAP50.pth"
train.output_dir = "/netscratch/naeem/phenobench_deta_syn_v3_augmented"

# max training iterations
train.max_iter = 3000
train.eval_period = 200
train.log_period = 10
train.checkpointer = dict(period=200, max_to_keep=2)

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
                        "/netscratch/naeem/sugarbeet_syn_v3/coco_annotations/instances_train.json",
                        # "/netscratch/naeem/sugarbeet_syn_v3/images")
                        # "/netscratch/naeem/sugarbeet_syn_v3/images_2")  # secogan output
                        # "/netscratch/naeem/sugarbeet_syn_v3/images_3")# LIS output Freestyle
                        "/netscratch/naeem/sugarbeet_syn_v3/images_augmented") # Albumentations output
register_coco_instances("pheno_val", {},
                        "/netscratch/naeem/phenobench/coco_annotations/coco_plants_panoptic_val.json",
                        "/netscratch/naeem/phenobench/val/")

