#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import sys
import time
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.file_io import PathManager
from detectron2.checkpoint import DetectionCheckpointer
from detrex.utils import WandbWriter
from detrex.modeling import ema
from glob import glob
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

CLASSES = [
    'crop', 'weed'
]
      
def generate_submission(model, predictor, image_root_dir=None):

    def toYOLO(boxes):
        pass

    
def do_test(cfg, model, eval_only=False):
    logger = logging.getLogger("detectron2")

    if eval_only:
        logger.info("Run evaluation under eval-only mode")
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            logger.info("Run evaluation with EMA.")
        else:
            logger.info("Run evaluation without EMA.")
        if "evaluator" in cfg.dataloader:
            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
            )
            print_csv_format(ret)
        return ret
    
    logger.info("Run evaluation without EMA.")
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)

        if cfg.train.model_ema.enabled:
            logger.info("Run evaluation with EMA.")
            with ema.apply_model_ema_and_restore(model):
                if "evaluator" in cfg.dataloader:
                    ema_ret = inference_on_dataset(
                        model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
                    )
                    print_csv_format(ema_ret)
                    ret.update(ema_ret)
        return ret


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    
    # using ema for evaluation
    ema.may_build_model_ema(cfg, model)
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone.res5[-1].register_forward_hook(
            lambda self, input, output: conv_features.append(output[0])
        ),
        # model.input_proj.register_forward_hook(
        #     lambda self, input, output: conv_features.append(output[0])
        # ),
        # # model.transformer.encoder.layers[-5].register_forward_hook(
        # #     lambda self, input, output: enc_attn_weights.append(output)
        # # ),
        # model.transformer.encoder.layers[-6].register_forward_hook(
        #     lambda self, input, output: enc_attn_weights.append(output)
        # ),
        # # model.transformer.decoder.layers[-5].register_forward_hook(
        # #     lambda self, input, output: dec_attn_weights.append(output)
        # # ),
        # model.transformer.decoder.layers[-1].ffns[0].register_forward_hook(
        #     lambda self, input, output: dec_attn_weights.append(output)
        # ),
    ]

    DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(cfg.train.init_checkpoint)
    model.eval()
    pred = {}
    images = glob(os.path.join('/mnt/e/datasets/phenobench/test/images', '*.png'))
    img_data = []
    for img, c in zip(images, range(len(images))):
        img_dict = {}
        img_dict['filename'] = img
        img = cv2.imread(img)
        pil_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(pil_image)
        # img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)
        orig = img.copy()
        img = np.transpose(img, (2, 0, 1))
        img_dict['height'] = img.shape[1]
        img_dict['width'] = img.shape[2]
        img_dict['image_id'] = c
        img_dict['image'] = torch.tensor(img, device='cpu', dtype=torch.uint8)
        img_data.append(img_dict)
        outputs = model(img_data)
        # don't need the list anymore
        conv_features = conv_features[0]
        enc_attn_weights = model.transformer.encoder.self_attention_maps
        dec_attn_weights = model.transformer.decoder.cross_attention_maps
        a = dec_attn_weights[5]
        a = torch.mean(a, dim=2, keepdim=False)
        a = a.reshape(a.shape[1:])
        a = a.detach().cpu().numpy()
        for idx, tmp in enumerate(a):
            plt.imshow(a[idx, : ,:])
        predictions = []
        for output in outputs:
            boxes = output['instances']._fields['pred_boxes'].tensor.cpu().detach().numpy()
            scores = output['instances']._fields['scores'].cpu().detach().numpy()
            labels = output['instances']._fields['pred_classes'].cpu().detach().numpy()
            indices = np.where(scores > 0.3)

            boxes = boxes[indices]
            scores = scores[indices]
            labels = labels[indices]

            filename = os.path.basename(img_dict['filename'])
            for box in boxes:
                cv2.rectangle(orig, 
                              (int(box[0]),int(box[1])),
                              (int(box[2]),int(box[3])),
                              color=(0, 255, 0),
                              thickness=2
                )
            cv2.imshow('image.png', orig)
            cv2.waitKey(0)
            # fig, axs = plt.subplots(ncols=len(indices[0]), nrows=2, figsize=(50, 14))
            # colors = COLORS * 100
            # h, w = conv_features.shape[1:]
            # focus_points = []
            # for idx in indices[0]:
            #     xmin, ymin, xmax, ymax = boxes[idx, 0], boxes[idx, 1], boxes[idx, 2], boxes[idx, 3] 
            #     focus_points.append((
            #                         int((ymax + ymin).item()/2),
            #                         int((xmax + xmin).item()/2), 
            #                         ))
            # for dec_attn_head_idx in range(len(dec_attn_weights)):
            #     for idx, ax_i in zip(indices[0], axs.T):
            #         xmin, ymin, xmax, ymax = boxes[idx, 0], boxes[idx, 1], boxes[idx, 2], boxes[idx, 3] 
            #         ax = ax_i[0]
            #         # for dec_attn_head_idx in range(len(dec_attn_weights)):
            #         ax.imshow(dec_attn_weights[dec_attn_head_idx][0, idx, :].detach().cpu().view(h,w))
            #         ax.axis('off')
            #         ax.set_title(f'query id: {idx.item()}')
            #         ax = ax_i[1]
            #         ax.imshow(pil_image)
            #         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
            #                                 fill=False, color='blue', linewidth=3))
            #         ax.axis('off')
            #         ax.set_title(CLASSES[labels[idx]])
            #     plt.savefig("dec_attention_weights_head_{}.png".format(dec_attn_head_idx), dpi=200)
            # # plt.show()
            # fig.tight_layout()
            # for query_idx in range(dec_attn_weights.shape[0]):
            #     sz = int(np.sqrt(dec_attn_weights.shape[1]))
            #     tmp = dec_attn_weights[query_idx, :].view(sz, sz)
            #     tmp = np.resize(tmp.detach().cpu().numpy(), (orig.shape[0], orig.shape[1]))
            #     a = ((tmp + np.abs(tmp.min())) / tmp.max()) * 255
            #     a = a.astype(np.uint8)
            #     th = cv2.threshold(tmp,tmp.min(),tmp.max(),cv2.THRESH_BINARY)[1]
            #     a = th / tmp.max()
            #     a = (a * 255).astype(np.uint8)
            #     blur = cv2.GaussianBlur(th,(13,13), 11)
            #     heatmap_img = cv2.applyColorMap(a, cv2.COLORMAP_JET)
            #     super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, orig, 0.5, 0)
            #     cv2.imwrite('image.png', super_imposed_img)
            # # cv2.imshow('prediction', super_imposed_img)
            # # cv2.waitKey()
            # cv2.imwrite('image.png', super_imposed_img)
                # output of the CNN
            print("Encoder attention:      ", enc_attn_weights[0].shape)
            print("Feature map:            ", conv_features.shape)

            # get the HxW shape of the feature maps of the CNN
            shape = conv_features.shape[1:]
            # and reshape the self-attention to a more interpretable shape
            sattn = enc_attn_weights[0].reshape(shape + shape).detach().cpu()
            print("Reshaped self-attention:", sattn.shape)

            # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
            fact = 32

            # let's select 4 reference points for visualization
            # idxs = [(200, 200), (280, 400), (200, 600), (440, 800),]
            # idxs = [(426, 260), (440, 426), (960, 77), (482, 882),]
            # idxs = [(260, 426), (426, 440), (77, 960), (882, 482),]
            idxs = focus_points
            # here we create the canvas
            fig = plt.figure(constrained_layout=True, figsize=(27, 10))
            # and we add one plot per reference point
            gs = fig.add_gridspec(len(focus_points)//2, len(idxs))
            axs = []
            if len(focus_points) % 2 == 0:
                for idx in range(len(focus_points)//2):
                    axs.append(fig.add_subplot(gs[idx, 0]))
                for idx in range(len(focus_points)//2):
                    axs.append(fig.add_subplot(gs[idx, -1]))
            else:
                for idx in range(int(len(focus_points)/2)):
                    axs.append(fig.add_subplot(gs[idx, 0]))
                for idx in range(int(len(focus_points)/2)-1):
                    axs.append(fig.add_subplot(gs[idx, -1]))
                
            # for each one of the reference points, let's plot the self-attention
            # for that point
            for idx_o, ax in zip(idxs, axs):
                idx = (idx_o[0] // fact, idx_o[1] // fact)
                # if idx[0] > 24:
                #     idx[0] == 24
                # if idx[1] > 24:
                #     idx[1] == 24
                ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
                ax.axis('off')
                ax.set_title(f'self-attention{idx_o}')

            # and now let's add the central image, with the reference points as red circles
            fcenter_ax = fig.add_subplot(gs[:, 1:-1])
            fcenter_ax.imshow(pil_image)
            for (y, x) in idxs:
                scale = pil_image.height / pil_image.size[1]
                x_proj = ((x // fact) + 0.5) * fact
                y_proj = ((y // fact) + 0.5) * fact
                fcenter_ax.add_patch(plt.Circle((x_proj * scale, y_proj * scale), 2.0, color='r'))
                fcenter_ax.text(x=x_proj * scale, 
                                y=(y_proj * scale) + 5, 
                                s='{}, {}'.format(y,x), 
                                color='w')
                fcenter_ax.axis('off')
            # plt.show()
            fig.tight_layout()
            plt.savefig("self_attention_maps", dpi=200)
            sys.exit()
            # plt.show()
            


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
