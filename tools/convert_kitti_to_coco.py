from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import cv2
from os import path
import os

from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d


'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving imagse boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''

def _bbox_to_coco_bbox(bbox):
    return [
        bbox[0], bbox[1],
        (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
    ]

def read_clib(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 2:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib


    


def convert(ann_dir, img_dir, calib_dir, split_dir, output_ann_dir, cat_ids, cat_info, label_3d=False, debug=False):
    splits = ['val', 'train']    
   
    for split in splits:
        print(f'Convert data for {split}:')
        ret = {'images': [], 'annotations': [], "categories": cat_info}
        image_set = open(path.join(split_dir, '{}.txt'.format(split)), 'r')
        
        for line in image_set:
            line = line[:-1] if line[-1]=='\n' else line
            image_id = int(line) 
            
            image_info = {
                'file_name': '{}.png'.format(line),
                'id': int(image_id),
            }
            if label_3d:
                calib_path = path.join(calib_dir, '{}.txt'.format(line))
                calib = read_clib(calib_path)
                image_info['calib'] = calib.tolist()
            
            ret['images'].append(image_info)
            if split == 'test':
                continue
            if debug:
                image = cv2.imread(
                    path.join(img_dir, image_info['file_name'])
                )
            
            ann_path = path.join(ann_dir, '{}.txt'.format(line))
            anns = open(ann_path, 'r')
            
            for ann_ind, txt in enumerate(anns):
                tmp = txt[:-1].split(' ')
                cat_id = cat_ids[tmp[0]]
                truncated = int(float(tmp[1]))
                occluded = int(tmp[2])
                alpha = float(tmp[3])
                bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
                dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
                location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
                rotation_y = float(tmp[14])

                ann = {
                    'image_id': image_id,
                    'id': int(len(ret['annotations']) + 1),
                    'category_id': cat_id,
                    'bbox': _bbox_to_coco_bbox(bbox),
                    'truncated': truncated,
                    'occluded': occluded
                }
                
                if label_3d:
                    ann['location'] = location
                    ann['rotation_y'] = rotation_y
                    ann['alpha'] = alpha
                    ann['depth'] = location[2]
                    ann['dim'] = dim
                    
                ret['annotations'].append(ann)
                
                if debug and cat_id != 9: # 9: dont care
                    cv2.rectangle(
                        image, 
                        (int(bbox[0]), int(bbox[1])), 
                        (int(bbox[2]), int(bbox[3])), 
                        (0, 255, 0), 
                        2
                    )
                    
                    if label_3d:
                        box_3d = compute_box_3d(dim, location, rotation_y)
                        box_2d = project_to_image(box_3d, calib)
                        image = draw_box_3d(image, box_2d)
                    
                        x = (bbox[0] + bbox[2]) / 2
                        depth = np.array([location[2]], dtype=np.float32)
                        pt_2d = np.array([
                            (bbox[0] + bbox[2]) / 2, 
                            (bbox[1] + bbox[3]) / 2
                        ],dtype=np.float32)
                        pt_3d = unproject_2d_to_3d(pt_2d, depth, calib)
                        pt_3d[1] += dim[0] / 2
                        print('pt_3d', pt_3d)
                        print('location', location)
                
            if debug:
                cv2.imwrite('image.png', image)
                quit()
        
        if not debug:
            print("# images: ", len(ret['images']))
            print("# annotations: ", len(ret['annotations']))
            # import pdb; pdb.set_trace()
            out_path = path.join(output_ann_dir, '{}.json'.format(split))
            json.dump(ret, open(out_path, 'w'))
    
        

if __name__ == "__main__":
    
    data_root_path = '/home/ubuntu/MyCenterNet/data/kitti'
    ann_dir = path.join(data_root_path, 'training/label_2')
    img_dir = path.join(data_root_path, 'training/image_2')
    calib_dir = path.join(data_root_path, 'training/calib')
    
    split_dir = path.join(data_root_path, 'splits')
    output_ann_dir = path.join(data_root_path, 'annotations')
    if not path.isdir(output_ann_dir):
        os.mkdir(output_ann_dir)
     
    cats = [
        'Pedestrian', 'Car', 'Cyclist', 
        'Van', 'Truck',  'Person_sitting',
        'Tram', 'Misc', 'DontCare'
    ]
    cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}

    cat_info = []
    for i, cat in enumerate(cats):
        cat_info.append({'name': cat, 'id': i + 1})

    convert(
        ann_dir, img_dir, calib_dir, split_dir, output_ann_dir,
        cat_ids, cat_info,
        label_3d=False, debug=True
    )