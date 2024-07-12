from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler


def default_coco_scheduler(epochs=50, decay_epochs=40, warmup_epochs=0):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 16 batch size, using 1440000/16=90000
    total_steps_16bs = epochs * 7500
    decay_steps = decay_epochs * 7500
    warmup_steps = warmup_epochs * 7500
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[decay_steps, total_steps_16bs],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps_16bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )

def default_phenobench_scheduler(epochs=12, decay_epochs=9, warmup_epochs=0):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 16800 (1400x12)
    training images (~12 Phenobench epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 30 batch size, using 16800/30=560
    total_steps_30bs = epochs * 560
    decay_steps = decay_epochs * 560
    warmup_steps = warmup_epochs * 560
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[decay_steps, total_steps_30bs],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps_30bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )

def default_synthetic_scheduler(epochs=12, decay_epochs=9, warmup_epochs=0):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 12000 (1000x12)
    training images (~12 Phenobench epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 30 batch size, using 12000/30=400
    # total_steps_30bs = 4000
    # decay_steps = 600
    warmup_steps = 200
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.5, 0.1],
        milestones=[600, 1000, 1600],
        # values=[1.0, 0.75, 0.6, 0.1],
        # milestones=[200, 300, 600, 1200],    # the milestone for 1 has to be where warm up ends.
        # values=[1.0, 0.5, 0.1],
        # milestones=[200, 1000, 2000],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / 3000,
        warmup_method="linear",
        warmup_factor=0.001,
    )

def default_synthetic_1e_4():
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 12000 (1000x12)
    training images (~12 Phenobench epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 30 batch size, using 12000/30=400
    warmup_steps = 200
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.5, 0.1],
        milestones=[1000, 1200, 1800],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / 3000,
        warmup_method="linear",
        warmup_factor=0.001,
    )

def default_synthetic_2e_4():
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 12000 (1000x12)
    training images (~12 Phenobench epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 30 batch size, using 12000/30=400
    warmup_steps = 300
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.5, 0.1],
        milestones=[300, 700, 1000],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / 3000,
        warmup_method="linear",
        warmup_factor=0.001,
    )
# default scheduler for detr
lr_multiplier_12ep_10drop = default_coco_scheduler(12, 10, 0)
lr_multiplier_12ep_pheno = default_phenobench_scheduler(12, 9, 1)
lr_multiplier_12ep_synthetic = default_synthetic_scheduler(12, 8, 1)
lr_multiplier_12ep_synthetic_1e_4 = default_synthetic_1e_4()
lr_multiplier_12ep_synthetic_2e_4 = default_synthetic_2e_4()
lr_multiplier_12ep_8bs_scheduler = default_coco_scheduler(24, 20, 0)
