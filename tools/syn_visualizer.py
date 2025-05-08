import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import cv2
import json
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from scipy.ndimage import label
from copy import deepcopy
import argparse

def generate_random_colors(n):
    return np.random.rand(n, 3)

def load_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    return coco_data, annotations_by_image, categories

def is_box_fully_inside(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return (x1 >= x2 and y1 >= y2 and 
            x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2)

def get_box_area(box):
    return box[2] * box[3]

def should_highlight(ann, other_annotations, min_box_size):
    bbox = ann['bbox']
    category_id = ann['category_id']
    box_area = get_box_area(bbox)
    
    if box_area < min_box_size:
        return True
    
    is_inside_another = any(is_box_fully_inside(bbox, other_ann['bbox']) for other_ann in other_annotations if other_ann != ann)
    
    if category_id == 1:
        return is_inside_another
    elif category_id == 2:
        return is_inside_another and box_area < 400
    
    return False

def visualize_annotations(image, mask, annotations, min_box_size):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.imshow(image)
    ax1.set_title('Original Image with Bounding Boxes')
    ax1.axis('off')

    unique_instances = np.unique(mask)
    colors = generate_random_colors(len(unique_instances))
    colors[0] = [0, 0, 0]  # Set background color to black
    color_map = ListedColormap(colors)

    for ann in annotations:
        bbox = ann['bbox']
        category_id = ann['category_id']
        
        if category_id not in [1, 2]:
            continue
        
        if should_highlight(ann, annotations, min_box_size):
            color = 'yellow' if category_id == 1 else 'purple'
        else:
            color = 'green' if category_id == 1 else 'red'
        
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                         fill=False, edgecolor=color, linewidth=2)
        ax1.add_patch(rect)

    im = ax2.imshow(mask, cmap=color_map, interpolation='nearest')
    ax2.set_title('Instance Segmentation Mask')
    ax2.axis('off')

    plt.colorbar(im, ax=ax2, label='Instance ID')

    legend_elements = [
        plt.Line2D([0], [0], color='green', lw=2, label='Class 1'),
        plt.Line2D([0], [0], color='yellow', lw=2, label='Class 1 (highlighted)'),
        plt.Line2D([0], [0], color='red', lw=2, label='Class 2'),
        plt.Line2D([0], [0], color='purple', lw=2, label='Class 2 (highlighted)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

def process_and_filter_boxes(root, min_box_size=100, output_json_path=None, visualize=False):
    instance_segmentation_files = sorted(glob(os.path.join(root, 'instance_segmentation/*.npz')))
    images = sorted(glob(os.path.join(root, 'images/*.png')))

    # coco_json_path = os.path.join(root, 'coco_annotations/filtered_annotations.json') 
    coco_json_path = os.path.join('filtered_anns.json') 
    coco_data, annotations_by_image, categories = load_coco_annotations(coco_json_path)

    filtered_annotations = []
    filtered_image_ids = set()

    for img_path, mask_path in zip(images[199:], instance_segmentation_files[199:]):
        image_id = int(os.path.splitext(os.path.basename(img_path))[0])

        if image_id in annotations_by_image:
            annotations = annotations_by_image[image_id]
            
            for ann in annotations:
                category_id = ann['category_id']
                
                if category_id not in [1, 2]:
                    continue
                
                if not should_highlight(ann, annotations, min_box_size):
                    filtered_annotations.append(ann)
                    filtered_image_ids.add(image_id)

        if visualize:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = np.load(mask_path)['array']
            visualize_annotations(image, mask, annotations, min_box_size)

    if output_json_path:
        filtered_coco_data = deepcopy(coco_data)
        filtered_coco_data['annotations'] = filtered_annotations
        filtered_coco_data['images'] = [img for img in coco_data['images'] if img['id'] in filtered_image_ids]
        
        # Ensure all necessary attributes are present
        required_keys = ['info', 'licenses', 'images', 'annotations', 'categories']
        for key in required_keys:
            if key not in filtered_coco_data:
                filtered_coco_data[key] = coco_data.get(key, [])
        
        with open(output_json_path, 'w') as f:
            json.dump(filtered_coco_data, f)
        print(f"Filtered annotations saved to {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter COCO annotations and optionally visualize results.")
    parser.add_argument("root", help="Root directory of the dataset")
    parser.add_argument("--output", default="filtered_annotations.json", help="Output path for filtered JSON")
    parser.add_argument("--min-box-size", type=int, default=100, help="Minimum box size for filtering")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization of annotations")
    
    args = parser.parse_args()

    process_and_filter_boxes(args.root, args.min_box_size, visualize=args.visualize)