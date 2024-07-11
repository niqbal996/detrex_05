from detrex.config import get_config
from .models.deta_r50 import model
from .scheduler.coco_scheduler import lr_multiplier_12ep_10drop as lr_multiplier

# using the default optimizer and dataloader
dataloader = get_config("common/data/sugarbeets.py").dataloader
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "/netscratch/naeem/phenobench_deta_baseline"

# max training iterations
train.max_iter = 20000
train.eval_period = 1000
train.checkpointer.period = 1000

# set training devices
train.device = "cuda"
model.device = train.device

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 4
def register_sugarbeets():
    from detectron2.data.datasets import register_coco_instances

    register_coco_instances("maize_syn_v2_train", {},
                            "/mnt/d/datasets/Corn_syn_dataset/2022_GIL_Paper_Dataset_V2/coco_anns/instances_train_2022_2.json",
                            "/mnt/d/datasets/Corn_syn_dataset/2022_GIL_Paper_Dataset_V2/camera_main_camera/rect")
    register_coco_instances("maize_real_v2_val", {},
                            "/mnt/d/datasets/Corn_syn_dataset/2022_GIL_Paper_Dataset_V2/coco_anns/instances_val_2022.json",
                            "/mnt/d/datasets/Corn_syn_dataset/GIL_dataset/all_days/data")
    # register_coco_instances("maize_syn_v2_val", {},
    #                         "/media/niqbal/T7/datasets/maize_yolo/annotations/instances_val.json",
    #                         "/media/niqbal/T7/datasets/maize_yolo/obj_train_data")

