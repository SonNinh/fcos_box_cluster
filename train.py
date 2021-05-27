from os import path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.MyDataset import MyDataset, AttnDataset
from models.losses import FocalLoss
from models.resnet import get_centernet
from opts import Opts
from utils.trainval import Logger, load_model, save_model, set_device


class Trainer(object):
    def __init__(self, opt) -> None:
        super().__init__()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.train_dataloader = DataLoader(
            MyDataset(opt, 'train'), 
            batch_size=opt.batch_size, 
            shuffle=True, num_workers=opt.num_workers, 
        )
        self.val_dataloader = DataLoader(
            MyDataset(opt, 'val'), 
            batch_size=opt.batch_size, 
            shuffle=False, num_workers=opt.num_workers
        )
        
        # reduction = 'none' if opt.margin_weight else 'sum'
        # self.margin_loss1 = torch.nn.L1Loss(reduction='none')
        # self.margin_loss2 = torch.nn.L1Loss(reduction='none')
        self.margin_loss3 = torch.nn.L1Loss(reduction='none')
        # self.seg_loss1 = FocalLoss()
        # self.seg_loss2 = FocalLoss()
        self.seg_loss3 = FocalLoss()

        self.model = get_centernet(18, opt.heads, head_conv=opt.head_conv)
        self.optimizer = torch.optim.Adam(self.model.parameters(), opt.lr)
        self.start_epoch = 0
        self.best_loss = 1e7

        if opt.load_model != '':
            self.model, self.optimizer, self.start_epoch, self.best_loss = load_model(
                self.model, opt.load_model, self.optimizer, 
                opt.resume, opt.lr, opt.lr_step
            )
        self.start_epoch += 1
        set_device(
            self.model, self.optimizer, opt.gpus, 
            opt.chunk_sizes, self.device
        )

        self.logger = Logger()
        self.logger.read(opt.save_dir)


    def run_epoch(self, epoch, phase):
        if phase == 'train':
            print(f'Epoch: {epoch}')
            
            self.model.train()
            tqdm_bar = tqdm(self.train_dataloader, mininterval=2)
        else:
            self.model.eval()
            tqdm_bar = tqdm(self.val_dataloader, mininterval=2)

        n_batch = 0
        loss_stat = {
            'seg1': {
                'data':0,
                'line_style': 'r--'
            },
            'seg2': {
                'data':0,
                'line_style': 'g--'
            },
            'seg3': {
                'data':0,
                'line_style': 'b--'
            },
            'margin1': {
                'data':0,
                'line_style': 'r'
            },
            'margin2': {
                'data':0,
                'line_style': 'g'
            },
            'margin3': {
                'data':0,
                'line_style': 'b'
            },
            'total': {
                'data':0,
                'line_style': 'bo-'
            }
        }
        
        for batch in tqdm_bar:
            n_batch += 1

            # move data to GPU
            for k in batch:
                if k != 'img':
                    batch[k] = batch[k].to(
                        device=self.device, non_blocking=True, 
                        dtype=torch.float
                    )

            preds = self.model(batch['inp'])
            # B, C, H1, W1 = preds['margin1'].size()
            # preds['margin1'] = preds['margin1'] * batch['seg1'].unsqueeze(2).repeat(1, 1, 4, 1, 1).reshape(B, -1, H1, W1)
            # B, C, H2, W2 = preds['margin2'].size()
            # preds['margin2'] = preds['margin2'] * batch['seg2'].unsqueeze(2).repeat(1, 1, 4, 1, 1).reshape(B, -1, H2, W2)
            B, C, H3, W3 = preds['margin3'].size()
            # preds['margin3'] = preds['margin3'] * batch['seg3'].unsqueeze(2).repeat(1, 1, 4, 1, 1).reshape(B, -1, H3, W3)
            
            # mloss1 = self.margin_loss1(preds['margin1'], batch['margin1'])
            # mloss2 = self.margin_loss2(preds['margin2'], batch['margin2'])
            mloss3 = self.margin_loss3(preds['margin3'], batch['margin3'])
            # mloss1 = mloss1.reshape(B, 8, 4, H1, W1).sum(axis=2)
            # mloss2 = mloss2.reshape(B, 8, 4, H2, W2).sum(axis=2)
            mloss3 = mloss3.reshape(B, 8, 4, H3, W3).sum(axis=2)

            # score = preds['score'] # (B, num_class, H, W, 3, 1)

            if opt.margin_weight:            
                # mloss1 = (mloss1 * batch['seg1']).sum()/batch['seg1'].sum()
                # mloss2 = (mloss2 * batch['seg2']).sum()/batch['seg2'].sum()
                mloss3 = (mloss3 * batch['seg3']).sum()/batch['seg3'].sum()
            else:
                # mask1 = batch['seg1'].gt(0).float()
                # mask2 = batch['seg2'].gt(0).float()
                mask3 = batch['seg3'].gt(0).float()
                # mloss1 = (mloss1 * mask1).sum()/(mask1.sum())
                # mloss2 = (mloss2 * mask2).sum()/(mask2.sum())
                mloss3 = (mloss3 * mask3).sum()/(mask3.sum())
            

            # sloss1 = self.seg_loss1(preds['seg1'], batch['seg1'])
            # sloss2 = self.seg_loss2(preds['seg2'], batch['seg2'])
            sloss3 = self.seg_loss3(preds['seg3'], batch['seg3'])
            
            # w1 = 0.8
            # w2 = 1.
            # w3 = 1.2
            # total_loss = w1*mloss1 + w2*mloss2 + w3*mloss3 + w1*sloss1 + w2*sloss2 + w3*sloss3
            total_loss = mloss3 + sloss3
            
            if phase == "train":
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            
            
            # loss_stat['seg1']['data'] += sloss1.item()
            # loss_stat['seg2']['data'] += sloss2.item()
            loss_stat['seg3']['data'] += sloss3.item()
            # loss_stat['margin1']['data'] += mloss1.item()
            # loss_stat['margin2']['data'] += mloss2.item()
            loss_stat['margin3']['data'] += mloss3.item()
            loss_stat['total']['data'] += total_loss.item()
            
            tqdm_bar.set_description(
                '---- total_loss: {:2.4}' 
                # '- seg1: {:2.4f} - seg2: {:2.4f}'
                ' - seg3: {:2.4f}'
                # ' - margin1: {:3.4f} - margin2: {:3.4f}' 
                ' - margin3: {:3.4f}'
                .format(
                    loss_stat['total']['data']/n_batch, 
                    # loss_stat['seg1']['data']/n_batch,
                    # loss_stat['seg2']['data']/n_batch,
                    loss_stat['seg3']['data']/n_batch,
                    # loss_stat['margin1']['data']/n_batch,
                    # loss_stat['margin2']['data']/n_batch,
                    loss_stat['margin3']['data']/n_batch
                )
            )
            
        
        for k in loss_stat:
            loss_stat[k]['data'] /= n_batch

        self.logger.log(loss_stat, phase, epoch)
        
        return loss_stat


    def __call__ (self):
        for epoch in range(self.start_epoch, self.start_epoch+opt.num_epochs):
            self.run_epoch(epoch, 'train')
            
            if epoch % opt.val_intervals == 0:
                loss_stat = self.run_epoch(epoch, 'val')
                
                if loss_stat[opt.metric]['data'] < self.best_loss:
                    self.best_loss = loss_stat[opt.metric]['data']
                    save_model(
                        path.join(opt.save_dir, 'best.pth'), 
                        epoch, self.best_loss, self.model, self.optimizer
                    )
                
                    self.logger.write(opt.save_dir)


class TrainerAttn(object):
    def __init__(self, opt) -> None:
        super().__init__()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.train_dataloader = DataLoader(
            AttnDataset(opt, 'train'), 
            batch_size=opt.batch_size, 
            shuffle=True, num_workers=opt.num_workers, 
        )
        self.val_dataloader = DataLoader(
            AttnDataset(opt, 'val'), 
            batch_size=opt.batch_size, 
            shuffle=False, num_workers=opt.num_workers
        )
        
        # reduction = 'none' if opt.margin_weight else 'sum'
        # self.margin_loss1 = torch.nn.L1Loss(reduction='none')
        # self.margin_loss2 = torch.nn.L1Loss(reduction='none')
        self.margin_loss3 = torch.nn.L1Loss(reduction='none')
        # self.seg_loss1 = FocalLoss()
        # self.seg_loss2 = FocalLoss()
        self.seg_loss3 = FocalLoss()

        self.model = get_centernet(18, opt.heads, head_conv=opt.head_conv)
        self.optimizer = torch.optim.Adam(self.model.parameters(), opt.lr)
        self.start_epoch = 0
        self.best_loss = 1e7

        if opt.load_model != '':
            self.model, self.optimizer, self.start_epoch, self.best_loss = load_model(
                self.model, opt.load_model, self.optimizer, 
                opt.resume, opt.lr, opt.lr_step
            )
        self.start_epoch += 1
        set_device(
            self.model, self.optimizer, opt.gpus, 
            opt.chunk_sizes, self.device
        )

        self.logger = Logger()
        self.logger.read(opt.save_dir)


    def run_epoch(self, epoch, phase):
        if phase == 'train':
            print(f'Epoch: {epoch}')
            
            self.model.train()
            tqdm_bar = tqdm(self.train_dataloader, mininterval=2)
        else:
            self.model.eval()
            torch.cuda.empty_cache()
            tqdm_bar = tqdm(self.val_dataloader, mininterval=2)

        n_batch = 0
        loss_stat = {
            'seg1': {
                'data':0,
                'line_style': 'r--'
            },
            'seg2': {
                'data':0,
                'line_style': 'g--'
            },
            'seg3': {
                'data':0,
                'line_style': 'b--'
            },
            'margin1': {
                'data':0,
                'line_style': 'r'
            },
            'margin2': {
                'data':0,
                'line_style': 'g'
            },
            'margin3': {
                'data':0,
                'line_style': 'b'
            },
            'total': {
                'data':0,
                'line_style': 'bo-'
            }
        }
        
        for batch in tqdm_bar:
            n_batch += 1

            # move data to GPU
            for k in batch:
                if k != 'img':
                    batch[k] = batch[k].to(
                        device=self.device, non_blocking=True, 
                        dtype=torch.float
                    )

            preds = self.model(batch['inp'], batch['margin3'], batch['mask3'], phase)
            B, C, H3, W3 = preds['margin3'].size()

            pred_margin3 = preds['margin3'].view(B, 8, 4, H3, W3)
            pred_margin3 = pred_margin3.unsqueeze(2).repeat(1, 1, 3, 1, 1, 1) # (B, n_class, num_gt, 4, h, w)
            mloss3 = self.margin_loss3(pred_margin3, batch['margin3']) # (B, n_class, num_gt, 4, h, w)
            mloss3 = mloss3.mean(axis=3) # (B, n_class, num_gt, h, w)
            mloss3 = mloss3.permute(0, 1, 3, 4, 2) # (B, n_class, h, w, num_gt)
            score = preds['score'] # (B, n_class, H, W, num_gt)
            mloss3 = torch.mul(mloss3, score).mean(-1)
            pos_mask3 = batch['seg3'].gt(0).float()
            mloss3 = (mloss3 * pos_mask3).sum()/(pos_mask3.sum())

            sloss3 = self.seg_loss3(preds['seg3'], batch['seg3'])

            total_loss = mloss3 + sloss3
            
            if phase == "train":
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            
            
            # loss_stat['seg1']['data'] += sloss1.item()
            # loss_stat['seg2']['data'] += sloss2.item()
            loss_stat['seg3']['data'] += sloss3.item()
            # loss_stat['margin1']['data'] += mloss1.item()
            # loss_stat['margin2']['data'] += mloss2.item()
            loss_stat['margin3']['data'] += mloss3.item()
            loss_stat['total']['data'] += total_loss.item()
            
            tqdm_bar.set_description(
                '---- total_loss: {:2.4}' 
                # '- seg1: {:2.4f} - seg2: {:2.4f}'
                ' - seg3: {:2.4f}'
                # ' - margin1: {:3.4f} - margin2: {:3.4f}' 
                ' - margin3: {:3.4f}'
                .format(
                    loss_stat['total']['data']/n_batch, 
                    # loss_stat['seg1']['data']/n_batch,
                    # loss_stat['seg2']['data']/n_batch,
                    loss_stat['seg3']['data']/n_batch,
                    # loss_stat['margin1']['data']/n_batch,
                    # loss_stat['margin2']['data']/n_batch,
                    loss_stat['margin3']['data']/n_batch
                )
            )
            
        
        for k in loss_stat:
            loss_stat[k]['data'] /= n_batch

        self.logger.log(loss_stat, phase, epoch)
        
        return loss_stat


    def __call__ (self):
        for epoch in range(self.start_epoch, self.start_epoch+opt.num_epochs):
            self.run_epoch(epoch, 'train')
            
            if epoch % opt.val_intervals == 0:
                loss_stat = self.run_epoch(epoch, 'val')
                
                if loss_stat[opt.metric]['data'] < self.best_loss:
                    self.best_loss = loss_stat[opt.metric]['data']
                    save_model(
                        path.join(opt.save_dir, 'best.pth'), 
                        epoch, self.best_loss, self.model, self.optimizer
                    )
                
                    self.logger.write(opt.save_dir)


if __name__ == "__main__":
    opt = Opts().init()
    trainer = TrainerAttn(opt)
    trainer()
