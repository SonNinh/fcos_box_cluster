import os
import math

import cv2
import numpy as np
from numpy.lib.shape_base import _make_along_axis_idx
import pycocotools.coco as coco
import torch
from opts import Opts
from torch.utils.data import Dataset
from utils.image import affine_transform, get_affine_transform, transform_preds


class MyDataset(Dataset):
    def __init__(self, opt, split):
        super(MyDataset, self).__init__()
        self.opt = opt
        self.img_dir = os.path.join(opt.data_dir, 'training/image_2')
        
        self.annot_path = os.path.join(
            opt.data_dir, 'annotations', '{}.json'.format(split if split=='train' else 'val')
        )
        self.max_objs = 50
        self.class_name = [
            '__background__', 'Pedestrian', 'Car', 'Cyclist',
            'Van', 'Truck',  'Person_sitting',
            'Tram', 'Misc', 'DontCare'
        ]
        self.cat_ids = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8}
        
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        self.split = split

        print('Loaded {} {} samples'.format(split, self.num_samples))
       
    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        img = cv2.imread(img_path)
        
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = self.opt.input_h, self.opt.input_w


        trans_input = get_affine_transform(
            c, s, [input_w, input_h]
        )
        img = cv2.warpAffine(
            img, trans_input,
            (input_w, input_h),
            flags=cv2.INTER_LINEAR
        )
        inp = (img.astype(np.float32) / 255.)
        inp = (inp - self.opt.mean) / self.opt.std
        inp = inp.transpose(2, 0, 1)

        sorted_id = [
            i for _, i in sorted(
                zip(anns, range(len(anns))), 
                key=lambda x: np.prod(x[0]['bbox'][2:4]),
                reverse=False
            )
        ][:num_objs]
        # sorted_id = [
        #     i for _, i in sorted(
        #         zip(anns, range(len(anns))), 
        #         key=lambda x: self._coco_box_to_bbox(x[0]['bbox'])[3],
        #         reverse=True
        #     )
        # ][-num_objs:]
        

        margin1, seg1 = self.generate_output(
            [input_h, input_w],
            c, s, 16, anns, sorted_id
        )
        margin2, seg2 = self.generate_output(
            [input_h, input_w],
            c, s, 8, anns, sorted_id
        )
        margin3, seg3 = self.generate_output(
            [input_h, input_w],
            c, s, 4, anns, sorted_id
        )
        
        ret = {
            'inp': inp, 
            'margin1': margin1, 
            'margin2': margin2, 
            'margin3': margin3, 
            'seg1': seg1, 
            'seg2': seg2, 
            'seg3': seg3
        }
        if self.split == 'test':
            ret['img'] = img

        return ret

    def generate_output(self, input_shape, center, scale, model_stride, anns, area_decreased_id):
        output_h = input_shape[0] // model_stride
        output_w = input_shape[1] // model_stride

        trans_output = get_affine_transform(
            center, scale,
            [output_w, output_h]
        )

        return self.helper(
            anns, trans_output, output_w, output_h, area_decreased_id
        )

    def helper(self, anns, trans_output, output_w, output_h, area_decreased_id):
        margin = np.zeros((4*(self.opt.num_classes-1), output_h, output_w), dtype=np.float32)
        seg = np.zeros((self.opt.num_classes-1, output_h, output_w), dtype=np.float32)
        

        for k in area_decreased_id: 
            ann = anns[k]
            cls_id = int(self.cat_ids[ann['category_id']])
            if cls_id == 8 or ann['occluded'] == -1: 
                continue # not use class 'DontCare'
            
            # bbox on input image
            bbox = self._coco_box_to_bbox(ann['bbox'])
            # bbox on output heatmap
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            
            if h > 0 and w > 0:
                
                left, right = math.ceil(bbox[0]), math.floor(bbox[2])
                top, bottom = math.ceil(bbox[1]), math.floor(bbox[3])
                dx, dy = np.meshgrid(
                    np.arange(left, right, dtype=np.float32),
                    np.arange(top, bottom, dtype=np.float32)
                )
                dl = (dx - bbox[0]).clip(min=0)
                dt = (dy - bbox[1]).clip(min=0)
                dr = (bbox[2] - dx).clip(min=0)
                db = (bbox[3] - dy).clip(min=0)
                margin[cls_id*4+0, top: bottom, left: right] = dl
                margin[cls_id*4+1, top: bottom, left: right] = dt
                margin[cls_id*4+2, top: bottom, left: right] = dr
                margin[cls_id*4+3, top: bottom, left: right] = db
                seg[cls_id, top: bottom, left: right] = np.maximum(
                    seg[cls_id, top: bottom, left: right],
                    np.minimum(dl, dr) / np.maximum(dl, dr) * np.minimum(dt, db) / np.maximum(dt, db)
                )
               
        return margin, seg


    def gen_objectness(self, ):
        return

    def _coco_box_to_bbox(self, box):
        bbox = np.array([
            box[0], box[1], box[0] + box[2], box[1] + box[3]
            ], dtype=np.float32)
        return bbox



class AttnDataset(Dataset):
    def __init__(self, opt, split):
        super().__init__()
        self.opt = opt
        self.img_dir = os.path.join(opt.data_dir, 'training/image_2')
        
        self.annot_path = os.path.join(
            opt.data_dir, 'annotations', '{}.json'.format(split if split=='train' else 'val')
        )
        self.max_objs = 50
        self.class_name = [
            '__background__', 'Pedestrian', 'Car', 'Cyclist',
            'Van', 'Truck',  'Person_sitting',
            'Tram', 'Misc', 'DontCare'
        ]
        self.cat_ids = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8}
        
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        self.split = split

        print('Loaded {} {} samples'.format(split, self.num_samples))
       
    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        img = cv2.imread(img_path)
        
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = self.opt.input_h, self.opt.input_w


        trans_input = get_affine_transform(
            c, s, [input_w, input_h]
        )
        img = cv2.warpAffine(
            img, trans_input,
            (input_w, input_h),
            flags=cv2.INTER_LINEAR
        )
        inp = (img.astype(np.float32) / 255.)
        inp = (inp - self.opt.mean) / self.opt.std
        inp = inp.transpose(2, 0, 1)

        sorted_id = [
            i for _, i in sorted(
                zip(anns, range(len(anns))), 
                key=lambda x: np.prod(x[0]['bbox'][2:4]),
                reverse=False
            )
        ][:num_objs]
        # sorted_id = [
        #     i for _, i in sorted(
        #         zip(anns, range(len(anns))), 
        #         key=lambda x: self._coco_box_to_bbox(x[0]['bbox'])[3],
        #         reverse=True
        #     )
        # ][-num_objs:]
        
        
        # margin1, seg1 = self.generate_output(
        #     [input_h, input_w],
        #     c, s, 16, anns, sorted_id
        # )
        # margin2, seg2 = self.generate_output(
        #     [input_h, input_w],
        #     c, s, 8, anns, sorted_id
        # )
        margin3, seg3, mask3 = self.generate_output(
            [input_h, input_w],
            c, s, 4, anns, sorted_id
        )
        
        ret = {
            'inp': inp, 
            # 'margin1': margin1, 
            # 'margin2': margin2, 
            'margin3': margin3, 
            # 'seg1': seg1, 
            # 'seg2': seg2, 
            'seg3': seg3,
            'mask3': mask3
        }
        if self.split == 'test':
            ret['img'] = img

        return ret



    def generate_output(self, input_shape, center, scale, model_stride, anns, area_decreased_id):
        output_h = input_shape[0] // model_stride
        output_w = input_shape[1] // model_stride

        trans_output = get_affine_transform(
            center, scale,
            [output_w, output_h]
        )

        return self.helper(
            anns, trans_output, output_w, output_h, area_decreased_id
        )

    def helper(self, anns, trans_output, output_w, output_h, area_decreased_id):
        num_gt = 3
        # (N, 3, 4, H, W)
        margin = np.zeros(
            (self.opt.num_classes-1, num_gt+1, 4, output_h, output_w),
            dtype=np.float32
        )
        ind = np.zeros((self.opt.num_classes-1, output_h, output_w), dtype=np.int)
        seg = np.zeros((self.opt.num_classes-1, output_h, output_w), dtype=np.float32)
        mask = np.zeros((self.opt.num_classes-1, output_h, output_w, num_gt+1), dtype=np.int)

        for k in area_decreased_id: 
            ann = anns[k]
            cls_id = int(self.cat_ids[ann['category_id']])
            if cls_id == 8 or ann['occluded'] == -1: 
                continue # not use class 'DontCare'
            
            # bbox on input image
            bbox = self._coco_box_to_bbox(ann['bbox'])
            # bbox on output heatmap
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            
            if h > 0 and w > 0:
                
                left, right = math.ceil(bbox[0]), math.floor(bbox[2])
                top, bottom = math.ceil(bbox[1]), math.floor(bbox[3])
                dx, dy = np.meshgrid(
                    np.arange(left, right, dtype=np.int),
                    np.arange(top, bottom, dtype=np.int)
                )
                dl = (dx - bbox[0]).clip(min=0)
                dt = (dy - bbox[1]).clip(min=0)
                dr = (bbox[2] - dx).clip(min=0)
                db = (bbox[3] - dy).clip(min=0)
                margin[cls_id, ind[cls_id, top: bottom, left: right], 0, dy, dx] = dl
                margin[cls_id, ind[cls_id, top: bottom, left: right], 1, dy, dx] = dt
                margin[cls_id, ind[cls_id, top: bottom, left: right], 2, dy, dx] = dr
                margin[cls_id, ind[cls_id, top: bottom, left: right], 3, dy, dx] = db

                mask[cls_id, dy, dx, ind[cls_id, top: bottom, left: right]] = 1

                ind[cls_id, top: bottom, left: right] += 1
                ind = np.clip(ind, 0, num_gt)


                seg[cls_id, top: bottom, left: right] = np.maximum(
                    seg[cls_id, top: bottom, left: right],
                    np.minimum(dl, dr) / np.maximum(dl, dr) * np.minimum(dt, db) / np.maximum(dt, db)
                )
        


        return (
            margin[:, :num_gt], # (n_class, num_gt, 4, h, w)
            seg,                # (n_class, h, w)
            mask[..., :num_gt]  # (n_class, h, w, num_gt)
        )

    def gen_objectness(self, ):
        return

    def _coco_box_to_bbox(self, box):
        bbox = np.array([
            box[0], box[1], box[0] + box[2], box[1] + box[3]
            ], dtype=np.float32)
        return bbox


if __name__ == "__main__":

    opt = Opts().init({'2d_det'})
    val_dataset = AttnDataset(opt, 'test')

    for i in range(15):
        batch = val_dataset[i]
        img = batch['img']
        # margin1 = batch['margin1']
        # margin2 = batch['margin2']
        margin3 = batch['margin3']
        # seg1 = batch['seg1']
        # seg2 = batch['seg2']
        seg3 = batch['seg3']
        mask3 = batch['mask3']
        print(margin3.shape)
        
    #     seg3 = torch.tensor(seg3)
    #     pred = torch.ones_like(seg3)
    #     seg3[:, 20:30, 20:25] = 1.

    #     print(loss(pred, seg3))
    #     continue

        # cv2.imwrite('imag.png', x)
        # print(img.shape, margin1.shape, margin2.shape, margin3.shape)
        input_h = val_dataset.opt.input_h
        input_w = val_dataset.opt.input_w

        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        

        print(input_h, input_w)
        
        dx, dy = np.meshgrid(
            np.arange(margin3.shape[-1], dtype=np.float32),
            np.arange(margin3.shape[-2], dtype=np.float32)
        )
        bboxs3 = np.zeros_like(margin3)
        # bboxs3[0::4] = dx - margin3[0::4]
        # bboxs3[1::4] = dy - margin3[1::4]
        # bboxs3[2::4] = dx + margin3[2::4]
        # bboxs3[3::4] = dy + margin3[3::4]
        bboxs3[:, :, 0, ...] = dx - margin3[:, :, 0, ...]
        bboxs3[:, :, 1, ...] = dy - margin3[:, :, 1, ...]
        bboxs3[:, :, 2, ...] = dx + margin3[:, :, 2, ...]
        bboxs3[:, :, 3, ...] = dy + margin3[:, :, 3, ...]
        
        
        colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (120, 200, 0),
            (0, 200, 120)
        ]
        cv2.imwrite(f'seg_{i}.png', seg3[1]*255)
        for cls_id in range(8):
            # print(seg3[cls_id].shape)
            
            # bboxs = bboxs3[cls_id*4:cls_id*4+4].reshape(1, 4, -1)
            bboxs = bboxs3[cls_id].reshape(3, 4, -1)
            bboxs = np.transpose(bboxs, (2, 0, 1))#.reshape(-1, 4)
            seg = seg3[cls_id:cls_id+1].reshape(-1)
            seg = np.where(seg>0, True, False)
            mask = mask3[cls_id:cls_id+1].reshape(-1, 3)
            bboxs = bboxs[mask==1].reshape(-1, 4)
            # print('bboxs', bboxs.shape)
            # bboxs = bboxs[seg].reshape(-1, 4)
            bboxs[:, :2] = transform_preds(bboxs[:, :2], c, s, [input_w//4,input_h//4])
            bboxs[:, 2:] = transform_preds(bboxs[:, 2:], c, s, [input_w//4,input_h//4])
            bboxs = bboxs.astype(np.int)
            
            bboxs = np.unique(bboxs, axis=0)
            # print(bboxs)
            for box in bboxs:
                # print(box)
                cv2.rectangle(
                    img, 
                    (box[0], box[1]),
                    (box[2], box[3]),
                    colors[cls_id], 1
                )

        img_id = val_dataset.images[i]
        file_name = val_dataset.coco.loadImgs(ids=[img_id])[0]['file_name']
        print(file_name)
        ann_ids = val_dataset.coco.getAnnIds(imgIds=[img_id])
        anns = val_dataset.coco.loadAnns(ids=ann_ids)
    

        for ann in anns:
            box = val_dataset._coco_box_to_bbox(ann['bbox']).astype(int)
            
            # print(box)
            cv2.rectangle(
                img, 
                (box[0], box[1]),
                (box[2], box[3]),
                (0, 0, 255), 2
            )


        cv2.imwrite(f'img_{i}.png', img)
