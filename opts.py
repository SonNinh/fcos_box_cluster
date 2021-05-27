import argparse
import os
from os import path


class Opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # basic experiment setting
        self.parser.add_argument(
            'task', default='2d_det',
            help='2d_det | 3d_det'
        )
        self.parser.add_argument('--exp_name', default='default')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')
        self.parser.add_argument(
            '--resume', action='store_true',
             help='resume an experiment. '
                  'Reloaded the optimizer parameter and '
                  'set load_model to model_last.pth '
                  'in the exp dir if load_model is empty.'
        ) 
        self.parser.add_argument('--dataset', default='kitti',
                                 help='path to dataset')
        
        # system
        self.parser.add_argument('--gpus', default='0', 
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317, 
                                 help='random seed') # from CornerNet

        # log
        self.parser.add_argument('--print_iter', type=int, default=0, 
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--hide_data_time', action='store_true',
                                 help='not display time during training.')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save model to disk every 5 epochs.')
        self.parser.add_argument('--metric', default='total', 
                                 help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                                 help='visualization threshold.')
        self.parser.add_argument('--debugger_theme', default='white', 
                                 choices=['white', 'black'])

        # model
        self.parser.add_argument('--arch', default='res_18', 
                                 help='model architecture. Currently tested'
                                      'res_18 | res_101 | resdcn_18 | resdcn_101 |')
        self.parser.add_argument('--head_conv', type=int, default=64,
                                 help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '64 for resnets')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')
        
        # train
        self.parser.add_argument('--lr', type=float, default=1.25e-4, 
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='20,40',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=140,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                 help='batch size on the master gpu.')
        self.parser.add_argument('--val_intervals', type=int, default=3,
                                 help='number of epochs to run validation.')
     

       
        # ctdet
        self.parser.add_argument('--margin_weight', action='store_true',
                                 help='margin weight for margin heatmaps.')
       
        self.parser.add_argument('--root_dir', default='/home/ubuntu/fcos_box_cluster', 
                                 help='model architecture. Currently tested')
       
    
    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        ############ training hyper parameters
        opt.lr_step = list(map(int, opt.lr_step.split(',')))

        ############ dataset
        # opt.root_dir = '/home/ubuntu/MyCenterNet' #os.path.dirname(__file__)
        opt.data_dir = os.path.join(opt.root_dir, 'data', opt.dataset)
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        if not path.isdir(opt.exp_dir):
            os.mkdir(opt.exp_dir)
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_name)
        if not path.isdir(opt.save_dir):
            os.mkdir(opt.save_dir)
            print('The output will be saved to', opt.save_dir)
        else:
            print('The output will override ', opt.save_dir)
            while True:
                key = input('Do you want to continue? [y/n]')
                if key == 'y': break
                elif key == 'n': quit()


        ############ resume training
        if opt.resume:
            opt.load_model = os.path.join(opt.save_dir, 'best.pth')
            print(f'The training will resume from {opt.load_model}')
        
        ############ GPU
        opt.gpus = list(map(int, opt.gpus.split(',')))
        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', opt.chunk_sizes)


        return opt

    
    def update_dataset_info_and_set_heads(self, opt):
        opt.input_h, opt.input_w = opt.default_resolution
        opt.output_h1 = opt.input_h // opt.down_ratio
        opt.output_w1 = opt.input_w // opt.down_ratio
        opt.output_h2 = opt.input_h // opt.down_ratio // 2
        opt.output_w2 = opt.input_w // opt.down_ratio // 2
        opt.output_h3 = opt.input_h // opt.down_ratio // 4
        opt.output_w3 = opt.input_w // opt.down_ratio // 4
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res1 = max(opt.output_h1, opt.output_w1)
        opt.output_res2 = max(opt.output_h2, opt.output_w2)
        opt.output_res3 = max(opt.output_h3, opt.output_w3)
        
        print(opt.task)
        if opt.task == '2d_det':
            opt.heads = {
                'margin': (opt.num_classes-1)*4, 
                'seg': (opt.num_classes-1),
                # 'reg': 2
            }
        else:
            assert 0, 'task not defined!'

        return opt

    def init(self, args=''):
        default_dataset_info = {
            'kitti': {
                'default_resolution': [384, 1280], 
                'num_classes': 9, 
                'mean': [0.485, 0.456, 0.406], 
                'std': [0.229, 0.224, 0.225],
            },
        }
       
    
        opt = self.parse(args)
        for k, v in default_dataset_info[opt.dataset].items():
            setattr(opt, k, v)

        opt = self.update_dataset_info_and_set_heads(opt)
        return opt

def get_opts():
    return Opts().init(args={'2d_det'})

if __name__ == "__main__":
    opt = get_opts()
    print(opt)
