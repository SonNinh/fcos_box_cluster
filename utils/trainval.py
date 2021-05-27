
from __future__ import absolute_import, division, print_function

import torch
from models.parallel import MyDataParallel
import matplotlib.pyplot as plt
from os import path
import yaml


class Logger(object):
    def __init__(self) -> None:
        super().__init__()
        self.data = {
            'train':{},
            'val': {},
            'train_epoch': [],
            'val_epoch': []
        }
        self.y_lim = 0
    
    def log(self, stat, phase, epoch):
        # if phase == 'train':
        #     self.num_epoch += 1

        for k, v in stat.items():
            if k not in self.data[phase]: 
                self.data[phase][k] = {
                    'data': [],
                    'line_style': v['line_style']
                }
            self.data[phase][k]['data'].append(v['data'])
            self.y_lim = max(self.y_lim, v['data'])

        self.data[f'{phase}_epoch'].append(epoch)


    def read(self, path_read):
        file_dir = path.join(path_read, 'loss.yaml')
        if path.exists(file_dir):
            with open(file_dir) as file:
                self.data = yaml.load(file, Loader=yaml.FullLoader)
            # self.num_epoch = self.data['train_epoch']

            for phase in ['train', 'val']:
                for loss in self.data[phase].values():
                    self.y_lim = max(self.y_lim, max(loss['data']))
        else:
            print('Cannot find:', file_dir)


    def write(self, path_write):
        f, axes = plt.subplots(2, 1, figsize=(15,15))

        training_phase = ['train', 'val']
        for phase, axs in zip(training_phase, axes):
            axs.set_title(f'{phase} loss')
            axs.grid(True)
            axs.set_ylim(top=self.y_lim)
            axs.set_xlim(left=0, right=len(self.data['train_epoch']))
            axs.set_xlabel('Epochs')
            axs.set_ylabel('Loss')

            for k, v in self.data[phase].items():
                axs.plot(self.data[f'{phase}_epoch'], v['data'], v['line_style'], label=k)
                
            axs.legend()
        
        plt.savefig(path.join(path_write, 'loss.png'))


        with open(path.join(path_write, 'loss.yaml'), 'w') as file:
            yaml.dump(self.data, file)


        


def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
  
    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]

    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the '+\
            'pre-trained weight. Please make sure '+\
            'you have correctly specified --arch xxx '+\
            'or set the correct --num_classes for your own dataset.'

    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '\
                    'loaded shape{}. {}'.format(
                k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')

    if optimizer is not None:
        return model, optimizer, checkpoint['epoch'], checkpoint['loss']
    else:
        return model


def save_model(path_save, epoch, loss, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()


    data = {'epoch': epoch, 'loss':loss, 'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()

    torch.save(data, path_save)
    print('New model was saved to', path_save)


def set_device(model, optimizer, gpus, chunk_sizes, device):
    if len(gpus) > 1:
        model = MyDataParallel(
            model, device_ids=gpus, 
            chunk_sizes=chunk_sizes
        ).to(device)
    else:
        model = model.to(device)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device, non_blocking=True)

