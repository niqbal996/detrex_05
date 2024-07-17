import json
from os.path import join
from os.path import basename
from tqdm import tqdm
import random
SEED = 448
random.seed(SEED)
# with open('/mnt/d/datasets/phenobench/coco_annotations/plants_panoptic_val.json') as handle:
#     dataset = json.loads(handle.read())
root_dir = '/netscratch/naeem/sugarbeet_syn_v6/coco_annotations'
# root_dir = '/mnt/d/datasets/phenobench/coco_annotations/'
with open(join(root_dir, 'instances_train.json')) as handle:
    dataset = json.loads(handle.read())

all_files = []
for i in range(5000):
    all_files.append('{:04d}.png'.format(i))

random.shuffle(all_files)

for sampling_ratio in tqdm(range(90,110,10)):
    subset = None
    # with open('syn_list_{}.txt'.format(sampling_ratio), 'r') as f:
    #     subset_list = sorted([line.rstrip('\n') for line in f])
    subset_list = sorted(all_files[0:int((sampling_ratio/100)*len(all_files))])
    subset = dataset.copy()
    subset['images'] = []
    subset['annotations'] = []
    keep_idxs = []
    coco_img_ids = []
    for image,idx in zip(dataset['images'], range(len(dataset['images']))):
        if basename(image['file_name']) in subset_list:
            keep_idxs.append(idx)
            coco_img_ids.append(image['id'])
        else:
            continue
    del subset['images']
    subset['images'] = []
    subset['annotations'] = []
    for idx in keep_idxs:
        subset['images'].append(dataset['images'][idx])

    for ann in dataset['annotations']:
        if ann['image_id'] in coco_img_ids:
            subset['annotations'].append(ann)
    # for idx in coco_img_ids:
    #     subset['annotations'].append(dataset['annotations'][idx])
    with open(join(root_dir, 'instances_{}.json'.format(sampling_ratio)), 'w+') as fp:
        json.dump(subset, fp)
