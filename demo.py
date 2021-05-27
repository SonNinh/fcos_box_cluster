import os
from os import path

import cv2
from networkx.drawing.nx_pylab import draw
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.MyDataset import MyDataset
from models.resnet import get_centernet
from opts import Opts
from utils.image import transform_preds
from utils.trainval import load_model
from utils.post_processing import box_clutering, draw_bboxs


def concat_image():

    return 


def demo():
    opt = Opts().init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_dataset = MyDataset(opt, 'test')

    val_dataloader = DataLoader(
        val_dataset, batch_size=opt.batch_size, 
        shuffle=False, num_workers=opt.num_workers, 
    )

    model = get_centernet(18, opt.heads, head_conv=opt.head_conv)
    model = load_model(
        model, 
        path.join(opt.save_dir, 'best.pth')
    )
    model = model.to(device)
    model.eval()

    if not path.isdir(path.join('outputs', opt.exp_name)):
        os.mkdir(path.join('outputs', opt.exp_name))
        os.mkdir(path.join('outputs', opt.exp_name, 'raw'))
    
    id_image = 0
    for batch in val_dataloader:
        for k in batch:
            if k != 'img':
                batch[k] = batch[k].to(device=device, non_blocking=True, dtype=torch.float)
        
        preds = model(batch['inp'], None, None, 'test')
        seg3 = preds['seg3'].cpu().detach().numpy()
        margin3 = preds['margin3'].cpu().detach().numpy()
        image = batch['img'].detach().numpy()


        np.save(path.join('outputs', opt.exp_name, 'raw', f'seg_{id_image}.npy'), seg3)
        np.save(path.join('outputs', opt.exp_name, 'raw', f'margin_{id_image}.npy'), margin3)
        np.save(path.join('outputs', opt.exp_name, 'raw', f'img_{id_image}.npy'), image)
        id_image += 1

        
        # c = np.array([
        #     image.shape[2] / 2., 
        #     image.shape[1] / 2.
        # ], dtype=np.float32)
        # s = max(image.shape[2], image.shape[1]) * 1.0
        # input_h = val_dataset.opt.input_h
        # input_w = val_dataset.opt.input_w

        # B = margin3.shape[0]
        # dx, dy = np.meshgrid(
        #     np.arange(margin3.shape[3], dtype=np.float32),
        #     np.arange(margin3.shape[2], dtype=np.float32)
        # )
        
        # bboxs = np.zeros_like(margin3)
        # bboxs[:, 0::4] = dx - margin3[:, 0::4]
        # bboxs[:, 1::4] = dy - margin3[:, 1::4]
        # bboxs[:, 2::4] = dx + margin3[:, 2::4]
        # bboxs[:, 3::4] = dy + margin3[:, 3::4]

        # # print(bboxs.shape)
        
        # bboxs = bboxs[:, 4:8].reshape(B, 4, -1)
        # bboxs = np.transpose(bboxs, (0, 2, 1))

        
        # seg3_flat = np.where(seg3>0.5, True, False)
        # # for i, m in enumerate(seg3[:, 1]):
        # #     cv2.imwrite(path.join('outputs', opt.exp_name, f'seg_{i}.png'), m*255)

        # seg3_flat = seg3_flat[:, 1:2].reshape(B, -1)

        # for i, bbox in enumerate(bboxs):
        #     out_image = image[i]
        #     good_box = bbox[seg3_flat[i]]

        #     good_box[:, :2] = transform_preds(good_box[:, :2], c, s, [input_w//4,input_h//4])
        #     good_box[:, 2:] = transform_preds(good_box[:, 2:], c, s, [input_w//4,input_h//4])
        #     good_box = good_box.astype(np.int)
            
        #     draw_bboxs(out_image, good_box, (0, 255, 0))

        #     good_box = box_clutering(good_box)
        #     draw_bboxs(out_image, good_box, (0, 0, 255))

        #     # print(seg3[i, 1:2].shape)
        #     seg_img = np.repeat(
        #         np.expand_dims(seg3[i, 1], -1), 
        #         3, axis=-1
        #     )*255
            
        #     seg_img = cv2.resize(
        #         seg_img, 
        #         (out_image.shape[1], out_image.shape[0]), 
        #         # interpolation = cv2.INTER_AREA
        #     )
            
        #     out_image = np.concatenate((out_image, seg_img))
        #     print(id_image)
        #     cv2.imwrite(path.join('outputs', opt.exp_name, f'box_{id_image}.png'), out_image)
        #     id_image += 1
        
        if id_image > 10: break

if __name__ == "__main__":
    demo()
