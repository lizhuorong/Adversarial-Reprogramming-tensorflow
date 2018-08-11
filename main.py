# -*- coding:utf-8 -*-
# Created on Tue Aug  7 10:31:23 2018
# @author: Zhuorong Li  <lizr@zucc.edu.cn>

import os
import argparse

from model import Adversarial_Reprogramming as AR

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--network_name', dest='network_name', default='resnet_v2_50', help='vgg_16,resnet_v2_50,inception_v3')
parser.add_argument('--image_size', dest='image_size', type=int, default=224, help='size of input images')

""" Arguments related to dataset"""
parser.add_argument('--dataset', dest='dataset', default='mnist', help='mnist, cifar')
parser.add_argument('--central_size', dest='central_size', type=int, default=28, help='28 for MNIST,32 for CIFAR10')

"""Arguments related to run mode"""
#parser.add_argument('--restore', dest='restore', default=None, action='store', type=int, help='Specify checkpoint id to restore.')

"""Arguments related to training"""
parser.add_argument('--max_epoch', dest='max_epoch', type=int, default=50, help='max num of epoch')
parser.add_argument('--lr', dest='lr', type=float, default=0.05, help='initial learning rate for adam')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=50, help='# images in batch')
parser.add_argument('--lmd', dest='lambda', type=float, default=0.0005, help='# weights of norm penalty')#0.01
parser.add_argument('--decay', type=float, default=0.96, help='Decay to apply to lr')

"""Arguments related to monitoring and outputs"""
parser.add_argument('--save_freq', dest='save_freq', type=int, default=5, help='save the model every save_freq sgd iterations')
parser.add_argument('--train_dir', dest='train_dir', default='./train', help='train logs are saved here')
parser.add_argument('--model_dir', dest='model_dir', default='./model', help='pretrained models are saved here')
parser.add_argument('--data_dir', dest='data_dir', default='./datasets', help='datasets are stored here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='datasets are stored here')

args = parser.parse_args()
print(args)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
    if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
   
    model = AR(args)
    model.run()

if __name__ == '__main__':
    main()    

    


