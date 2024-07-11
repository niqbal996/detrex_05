from glob import glob
import json
import os
import numpy as np
from copy import deepcopy

with open('/netscratch/naeem/sugarbeet_syn_v2/coco_annotations/instances_train.json', 'r') as f:
    data = json.load(f)
total_anns = len(data['annotations']) * 3
new_data = deepcopy(data)
for factor in range(1,3):
    print('factor: {}'.format(factor))
    tmp_images = [] 
    tmp_annotations = []
    tmp_images = deepcopy(data['images'])
    tmp_annotations = deepcopy(data['annotations'])
    for file, i in zip(tmp_images, range(len(tmp_images))):
        name = file['file_name']
        new_name = list(name)
        new_name[0] = '{}'.format(factor)
        new_name = "".join(new_name)
        tmp_images[i]['file_name'] = new_name
        tmp_images[i]['id'] = file['id'] + (factor*len(tmp_images))
    
    for ann, i in zip(tmp_annotations, range(len(tmp_annotations))):
        tmp_annotations[i]['image_id'] = int(ann['image_id'] + (factor*len(tmp_images)))
        tmp_annotations[i]['id'] = int(ann['id'] + (factor*len(tmp_annotations)))

    new_data['images'].extend(tmp_images)
    new_data['annotations'].extend(tmp_annotations)

with open('/netscratch/naeem/sugarbeet_syn_v2/coco_annotations/instances_trainx3.json', 'w') as f:
    json.dump(new_data, f, default=int, indent=4)
    
print('hold')