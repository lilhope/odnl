from __future__ import print_function
import argparse
import mxnet as mx
import os
from rcnn.config import config, default, generate_config
from rcnn.tools.test_rcnn import test_rcnn


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    args, rest = parser.parse_known_args()
    data_root = os.path.join(os.getcwd(),default.root_path)
    
    parser.add_argument('--root_path', help='output data folder', default=data_root, type=str)
    parser.add_argument('--subset',help='subset of dataset,only for refer dataset',default=default.subset,type=str)
    parser.add_argument('--split',help='split of dataset,only for refer dataset',default=default.split,type=str)
    # testing
    parser.add_argument('--prefix', help='model to test with', default=default.e2e_prefix, type=str)
    parser.add_argument('--epoch', help='model to test with', default=default.e2e_epoch, type=int)
    parser.add_argument('--gpu', help='GPU device to test with', default=0, type=int)
    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-1, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--has_rpn', help='generate proposals on the fly', action='store_true', default=True)
    parser.add_argument('--proposal', help='can be ss for selective search or rpn', default='rpn', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ctx = mx.gpu(args.gpu)
    print(args)
    test_rcnn(args.network, args.dataset, args.root_path, args.subset, args.split,
              ctx, args.prefix, args.epoch,
              args.vis, args.shuffle, args.has_rpn, args.proposal, args.thresh)

if __name__ == '__main__':
    main()
