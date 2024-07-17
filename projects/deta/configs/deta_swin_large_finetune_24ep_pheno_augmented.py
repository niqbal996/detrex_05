from detrex.config import get_config
from .models.deta_swin import model

# dataloader = get_config("common/data/sugarbeets.py").dataloader

from configs.common.data.sugarbeets_aug_subset import SubsetDataloader
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# 24ep for finetuning
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_24ep
train.init_checkpoint = "/netscratch/naeem/converted_deta_swin_o365_finetune.pth"
train.output_dir = "/netscratch/naeem/attention_maps_v6/dataset_size_experiments/phenobench_deta_swin_syn_v6_subset_{}".format(10)
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

train_batch_size = 8
train_workers = 8
test_batch_size = 8 
test_workers = 8


from detectron2.data.datasets import register_coco_instances
register_coco_instances("pheno_train", {},
                        "/netscratch/naeem/phenobench/coco_annotations/coco_plants_panoptic_train.json",
                        "/netscratch/naeem/phenobench/train/")
register_coco_instances("syn_pheno_train", {},
                        "/netscratch/naeem/sugarbeet_syn_v2/coco_annotations/instances_train.json",
                        "/netscratch/naeem/sugarbeet_syn_v2/images_augmented_2") # Albumentations output
dataloaders = []
for idx, factor in enumerate(range(10, 110, 10)):
    register_coco_instances("syn_pheno_train_{}".format(factor), {},
                        "/netscratch/naeem/sugarbeet_syn_v6/coco_annotations/instances_{}.json".format(factor),
                        "/netscratch/naeem/sugarbeet_syn_v6/images") # Albumentations output,
    dataloaders.append(SubsetDataloader(factor).dataloader)
    # modify dataloader config
    dataloaders[idx].train.total_batch_size = train_batch_size          # 30 for ResNet50 and 8 for SWIN backbone on 80GB 
    dataloaders[idx].train.num_workers = train_workers
    dataloaders[idx].test.batch_size = test_batch_size
    dataloaders[idx].test.num_workers = test_workers

dataloader = dataloaders[9]  # Use the full dataset by default. If args.subset_index != 100, then this will be changed later in train_net.py
register_coco_instances("pheno_val", {},
                        "/netscratch/naeem/phenobench/coco_annotations/coco_plants_panoptic_val.json",
                        "/netscratch/naeem/phenobench/val/")