import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import cv2
import numpy as np
import requests
import torch
import torchvision
from collections import OrderedDict
from PIL import Image
from pytorch_grad_cam import AblationCAM, EigenCAM, ScoreCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN, AblationLayer
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
from glob import glob
import os
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    default_argument_parser,
    default_setup
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.checkpoint import DetectionCheckpointer

class FasterRCNNBoxScoreTarget:
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    """
    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()
        if isinstance(model_outputs, dict) and len(model_outputs) == 1:
            boxes = model_outputs["instances"].get("pred_boxes").tensor
            labels = model_outputs["instances"].get("pred_classes")
            scores = model_outputs["instances"].get("scores")
        else:
            boxes = model_outputs["boxes"]
            labels = model_outputs["labels"]
            scores = model_outputs["scores"]
        if len(boxes) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = torchvision.ops.box_iou(box, boxes)
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and labels[index] == label:
                score = ious[0, index] + scores[index]
                output = output + score
        return output
    
class AblationLayerFasterRCNN(AblationLayer):
    def __init__(self):
        super(AblationLayerFasterRCNN, self).__init__()

    def set_next_batch(
            self,
            input_batch_index,
            activations,
            num_channels_to_ablate):
        """ Extract the next batch member from activations,
            and repeat it num_channels_to_ablate times.
        """
        self.activations = OrderedDict()
        for key, value in activations.items():
            fpn_activation = value[input_batch_index,
                                   :, :, :].clone().unsqueeze(0)
            self.activations[key] = fpn_activation.repeat(
                num_channels_to_ablate, 1, 1, 1)

    def __call__(self, x):
        result = self.activations
        # layers = {0: '0', 1: '1', 2: '2', 3: '3', 4: 'pool'}
        # num_channels_to_ablate = result['pool'].size(0)
        layers = {0: 'p2', 1: 'p3', 2: 'p4', 3: 'p5', 4: 'p6'}
        # layers = {0: 'p3', 1: 'p4', 2: 'p5', 3: 'p6'}
        num_channels_to_ablate = result['p6'].size(0)
        for i in range(num_channels_to_ablate):
            pyramid_layer = int(self.indices[i] / 256)
            index_in_pyramid_layer = int(self.indices[i] % 256)
            result[layers[pyramid_layer]][i,
                                          index_in_pyramid_layer, :, :] = -1000
        return result
    
def predict(input_tensor, model, device, detection_threshold, mode='torch'):
    if mode == 'torch':
        outputs = model(input_tensor)
        pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
        pred_labels = outputs[0]['labels'].cpu().numpy()
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        
        boxes, classes, labels, indices = [], [], [], []
        for index in range(len(pred_scores)):
            if pred_scores[index] >= detection_threshold:
                boxes.append(pred_bboxes[index].astype(np.int32))
                classes.append(pred_classes[index])
                labels.append(pred_labels[index])
                indices.append(index)
        boxes = np.int32(boxes)
        return boxes, classes, labels, indices, pred_scores
    else:
        outputs = model(input_tensor)
        for output in outputs:
            boxes = output['instances']._fields['pred_boxes'].tensor.cpu().detach().numpy()
            scores = output['instances']._fields['scores'].cpu().detach().numpy()
            labels = output['instances']._fields['pred_classes'].cpu().detach().numpy()
            classes = [pheno_names[i] for i in labels]
            indices = np.where(scores > detection_threshold)

            boxes = boxes[indices]
            scores = scores[indices]
            labels = labels[indices]
        return boxes, classes, labels, indices, scores

def draw_boxes(boxes, labels, classes, scores, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, f'{scores[i]:.4f}', 
                    # f'{classes[i]} {np.round(scores[i], 2):.2f}'
                (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                lineType=cv2.LINE_AA)
    return image

def renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    images = []
    for x1, y1, x2, y2 in boxes:
        img = renormalized_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
        images.append(img)
    
    renormalized_cam = np.max(np.float32(images), axis = 0)
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    image_with_bounding_boxes = draw_boxes(boxes, labels, classes, eigencam_image_renormalized)
    return image_with_bounding_boxes

def get_model(args, mode='torch', device='cpu'):
    if mode == 'torch':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        model.to(device)
    else:
        cfg = LazyConfig.load(args.config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
        default_setup(cfg, args)
        
        model = instantiate(cfg.model)
        model.to(device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        model.eval()
    return model

def get_sample_image(image_path='../sample.jpg', mode='local', device='cpu'):
    if mode == 'local':
        img_data = []
        img_dict = {}
        img_dict['filename'] = image_path
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
        image_float_np = np.float32(img) / 255
        orig = img.copy()
        img = np.transpose(img, (2, 0, 1))
        img_dict['height'] = img.shape[1]
        img_dict['width'] = img.shape[2]
        img_dict['image_id'] = 1
        img_dict['image'] = torch.tensor(img, device=device, dtype=torch.uint8)
        img_data.append(img_dict)
    else:
        image = Image.open(requests.get(image_path, stream=True).raw)
        image = np.array(image)
        image_float_np = np.float32(image) / 255
        orig = image.copy()
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        img_data = transform(image)
        img_data = img_data.to(device)
        # Add a batch dimension:
        img_data = img_data.unsqueeze(0)
    
    return img_data, orig, image_float_np

def renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    images = []
    boxes = boxes.astype(np.uint16)
    for x1, y1, x2, y2 in boxes:
        img = renormalized_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
        images.append(img)
    
    renormalized_cam = np.max(np.float32(images), axis = 0)
    # renormalized_cam = scale_cam_image(renormalized_cam)
    # eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    # image_with_bounding_boxes = draw_boxes(boxes, labels, classes, eigencam_image_renormalized)
    return renormalized_cam

def find_boxes_with_iou(input_box, boxes, iou_threshold=0.5):
    """
    Finds the indices of boxes that have an IoU greater than the given threshold with the input box.

    Args:
        input_box (numpy.ndarray or torch.Tensor): The input box of shape (4,) in [x1, y1, x2, y2] format.
        boxes (numpy.ndarray or torch.Tensor): A list of boxes of shape (N, 4) in [x1, y1, x2, y2] format.
        iou_threshold (float): The IoU threshold to filter boxes.

    Returns:
        list: Indices of boxes that satisfy the IoU threshold.
    """
    # Convert input_box and boxes to torch tensors if they are numpy arrays
    if isinstance(input_box, np.ndarray):
        input_box = torch.tensor(input_box, dtype=torch.float32)
    if isinstance(boxes, np.ndarray):
        boxes = torch.tensor(boxes, dtype=torch.float32)    
    # Compute IoU between the input box and all boxes
    ious = torchvision.ops.box_iou(input_box, boxes)

    # Find indices of boxes with IoU greater than the threshold
    indices = torch.where(ious[0] > iou_threshold)[0]

    return indices.tolist(), ious[0][indices].tolist()

coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', \
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
              'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
              'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
              'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']
pheno_names = ['sugarbeet', 'weed']
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
COLORS_PHENO = np.array([(255, 255, 255), (255, 255, 0)], dtype=np.float64)
def main(args):
    args = default_argument_parser().parse_args()
    # This will help us create a different color for each class
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_url = "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png"
    mode = 'detrex'
    # mode = 'torch'
    data_folder = '/mnt/e/datasets/explain_trial_set/single'
    # data_folder = '/mnt/e/datasets/sugarbeet_syn_v6/images'
    
    # data_folder = '/mnt/e/datasets/cropandweed_dataset/labelIds/SugarBeet1'
    # image_folder = '/mnt/e/datasets/cropandweed_dataset/images'
    # labels = glob(os.path.join(data_folder, '*.png'))
    
    images = glob(os.path.join(data_folder, '*.png'))
    model = get_model(args, mode=mode, device=device)
    target_layers = [model.backbone]
    # target_layers = [model.neck]
    cam = AblationCAM(model,
                target_layers, 
                #    use_cuda=torch.cuda.is_available(),
                reshape_transform=fasterrcnn_reshape_transform,
                ablation_layer=AblationLayerFasterRCNN(),
                ratio_channels_to_ablate=0.1,
                batch_size=1)
    # cam = EigenCAM(model,
    #             target_layers, 
    #             #    use_cuda=torch.cuda.is_available(),
    #             reshape_transform=fasterrcnn_reshape_transform)
    if mode == 'torch':
            sample, sample_orig, sample_np = get_sample_image(image_path=image_url, mode='url', device=device)
            boxes, classes, labels, indices, scores = predict(sample, model, device, 0.5, mode=mode)
            targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
            # CAM expects a batch of images, so we need to add a batch dimension with B X C X H X W
            grayscale_cam = cam(input_tensor=sample, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam = renormalize_cam_in_bounding_boxes(boxes, sample_np, grayscale_cam)
            cam_image = show_cam_on_image(sample_np, grayscale_cam, use_rgb=True)
            image_with_bounding_boxes = draw_boxes(boxes, labels, classes, scores, cam_image)
            cv2.imshow("Image", cv2.cvtColor(image_with_bounding_boxes, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
    else:
        for i, image_path in enumerate(images):
            sample, sample_orig, sample_np = get_sample_image(
                    image_path=image_path, 
                    # image_path=os.path.join(image_folder, os.path.basename(image_path)[:-4] + '.jpg'), 
                    mode='local', device=device)

            boxes, classes, labels, indices, scores = predict(sample, model, device, 0.5, mode=mode)
            image = draw_boxes(boxes, labels, classes, scores, sample_orig)
            # Show the image:
            cv2.imshow("Image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            box_gt = np.array([[381 , 907, 483, 981]], dtype=np.float32)
            indices, iou = find_boxes_with_iou(box_gt, boxes, iou_threshold=0.5)
            for i, index in enumerate(indices):
                print('GT_IOU: {} and Confidence value: {}'.format(iou[i], scores[index]))
                box = [boxes[index]]
                label = np.array([labels[index]], dtype=np.int32)
                score = np.array([scores[index]], dtype=np.float32)
                classes = ['sugarbeet']
                # tmp_labels, tmp_boxes = labels[0:1], boxes[0:1, :]
                targets = [FasterRCNNBoxScoreTarget(labels=label, bounding_boxes=box)]
                # CAM expects a batch of images, so we need to add a batch dimension with B X C X H X W
                grayscale_cam = cam(input_tensor=sample[0]['image'].unsqueeze(0), targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                # grayscale_cam = renormalize_cam_in_bounding_boxes(boxes, sample_np, grayscale_cam)
                cam_image = show_cam_on_image(sample_np, grayscale_cam, use_rgb=True)
                # image_with_bounding_boxes = draw_boxes(boxes, labels, classes, scores, cam_image)
                image_with_bounding_boxes = draw_boxes(box, label, classes, score, cam_image)
                # cv2.imwrite(os.path.join('/home/niqbal/git/aa_transformers/detrex/real_on_syn_images/sugarbeets', os.path.basename(image_path)[:-4] + '_cam.png'), 
                #             cv2.cvtColor(image_with_bounding_boxes, cv2.COLOR_RGB2BGR))
                # Show the image:
                cv2.imshow("Image", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)