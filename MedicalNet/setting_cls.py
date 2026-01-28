'''
Configs for PCR classification training & testing
'''

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_list',
        default='./data/train_cls.txt',
        type=str,
        help='Path for training image list file')
    parser.add_argument(
        '--val_list',
        default='./data/val_cls.txt',
        type=str,
        help='Path for validation image list file')
    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float,
        help='Initial learning rate')
    parser.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='Number of jobs')
    parser.add_argument(
        '--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument(
        '--phase', default='train', type=str, help='Phase of train or test')
    parser.add_argument(
        '--save_intervals',
        default=10,
        type=int,
        help='Interation for saving model')
    parser.add_argument(
        '--n_epochs',
        default=50,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--input_D',
        default=112,
        type=int,
        help='Input size of depth')
    parser.add_argument(
        '--input_H',
        default=112,
        type=int,
        help='Input size of height')
    parser.add_argument(
        '--input_W',
        default=112,
        type=int,
        help='Input size of width')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Path for resume model.')
    parser.add_argument(
        '--pretrain_path',
        default='',
        type=str,
        help='Path for pretrained model.')
    parser.add_argument(
        '--new_layer_name',
        default='fc',
        type=str,
        help='New layer name for fine-tune')
    parser.add_argument(
        '--num_classes',
        default=2,
        type=int,
        help='Number of classes')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,
        help='Gpu id lists')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet)')
    parser.add_argument(
        '--model_depth',
        default=50,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument(
        '--ci_test', action='store_true', help='If true, ci testing is used.')
    args = parser.parse_args()
    args.save_folder = "./trails/models/{}_{}_cls".format(args.model, args.model_depth)
    return args
