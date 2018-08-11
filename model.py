# -*- coding: utf-8 -*-
# Created on Tue Aug  7 10:29:59 2018
# @author: Zhuorong Li  <lizr@zucc.edu.cn>

# To get pre-trained models used in this repository:
# wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
# wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
# wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
# More pre-trained models can be found: https://github.com/tensorflow/models/tree/master/research/slim

# MNIST dataset will be automatically downloaded after running the scripts.


import numpy as np
import os
import scipy.misc
from time import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
from nets import vgg,inception,inception_v3,resnet_v2

mnist = input_data.read_data_sets('./datasets/MNIST_data/', one_hot=True) 

class Adversarial_Reprogramming(object):
    def __init__(self, sess, network_name='resnet_v2_50',image_size=224, dataset='mnist',\
                       central_size=28,max_epoch=50,lr=0.05,save_freq=5, \
                       batch_size=50,lmd=2e-6,decay=0.96,\
                       train_dir='./train', model_dir='./model',data_dir='./datasets',sample_dir='./sample'):
        
        self.network_name=network_name
        self.image_size=image_size
        self.dataset=dataset
        self.central_size=central_size
        self.max_epoch=max_epoch
        self.lr=lr
        self.batch_size=batch_size
        self.lmd=lmd
        self.decay=decay
        self.save_freq=save_freq
        self.train_dir=train_dir
        self.model_dir=model_dir
        self.data_dir=data_dir
        self.sample_dir=sample_dir

        
    def label_mapping(self):
            imagenet_label = np.zeros([1001,10])
            imagenet_label[0:10,0:10]=np.eye(10)
            return tf.constant(imagenet_label, dtype=tf.float32)  
    
    def adv_program(self,central_image):
        
        if self.dataset == 'mnist':
            self.central_size = 28
        else:
            self.central_size = 32
            
        if self.network_name.startswith('inception'):
            self.image_size = 299
            means = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        else:
            self.image_size = 224
            means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
        M = np.pad(np.zeros([1, self.central_size, self.central_size, 3]),\
                           [[0,0], [int((np.ceil(self.image_size/2.))-self.central_size/2.), int((np.floor(self.image_size/2.))-self.central_size/2.)],\
                            [int((np.ceil(self.image_size/2.))-self.central_size/2.), int((np.floor(self.image_size/2.))-self.central_size/2.)],\
                             [0,0]],'constant', constant_values = 1)
        self.M = tf.constant(M, dtype=tf.float32)
        with tf.variable_scope('adv_program',reuse=tf.AUTO_REUSE):
            self.W = tf.get_variable('program',shape=[1,self.image_size,self.image_size,3], dtype = tf.float32)
        if self.dataset == 'mnist':
           central_image  = tf.concat([central_image, central_image, central_image], axis = -1) 
        self.X = tf.pad(central_image,
					paddings = tf.constant([[0,0], [int((np.ceil(self.image_size/2.))-self.central_size/2.), int((np.floor(self.image_size/2.))-self.central_size/2.)],\
                             [int((np.ceil(self.image_size/2.))-self.central_size/2.), int((np.floor(self.image_size/2.))-self.central_size/2.)], [0,0]]))
        self.P = tf.nn.tanh(tf.multiply(self.W, self.M))
        self.X_adv = self.X + self.P
        
        self.channels = tf.split(self.X_adv, axis=3, num_or_size_splits=3)
        for i in range(3):
            self.channels[i] -= means[i]
            self.channels[i] /= std[i]
        self.X_adv = tf.concat(self.X_adv,axis=3) 
        
        return self.X_adv
    
    def run(self):
        
        if self.dataset == 'mnist':
            input_images  = tf.placeholder(shape = [None, 28,28,1], dtype = tf.float32)
        else:
            input_images  = tf.placeholder(shape = [None, self.image_size,self.image_size,3], dtype = tf.float32)
        Y = tf.placeholder(tf.float32, shape=[None, 10]) 
        
        ## load ImageNet classifier
        if self.network_name == 'resnet_v2_50':
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                self.imagenet_logits,_ = resnet_v2.resnet_v2_50(self.adv_program(input_images), num_classes = 1001,is_training=False)
                #print(self.imagenet_logits)#Tensor("resnet_v2_50/SpatialSqueeze:0", shape=(?, 1001), dtype=float32)
                self.disturbed_logits = tf.matmul(self.imagenet_logits,self.label_mapping())
                #print(self.disturbed_logits )#Tensor("MatMul:0", shape=(?, 10), dtype=float32)
                #print(self.label_mapping())#Tensor("Const_3:0", shape=(1001, 10), dtype=float32)
                init_fn = slim.assign_from_checkpoint_fn(os.path.join(self.model_dir,self.network_name+'.ckpt'),\
                                                         slim.get_model_variables('resnet_v2_50'))
        elif self.network_name == 'inception_v3':
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                self.imagenet_logits,_ = inception_v3.inception_v3(self.adv_program(input_images), num_classes = 1001,is_training=False)
                self.disturbed_logits = tf.matmul(self.imagenet_logits,self.label_mapping())
                init_fn = slim.assign_from_checkpoint_fn(os.path.join(self.model_dir,self.network_name+'.ckpt'),\
                                                         slim.get_model_variables('InceptionV3'))
        
        else:
            with slim.arg_scope(vgg.vgg_arg_scope()):
                self.imagenet_logits,_ = vgg.vgg_16(self.adv_program(input_images), num_classes = 1001,is_training=False)
                self.disturbed_logits = tf.matmul(self.imagenet_logits,self.label_mapping())
                init_fn = slim.assign_from_checkpoint_fn(os.path.join(self.model_dir,self.network_name+'.ckpt'),\
                                                         slim.get_model_variables('vgg_16'))
        
        ## loss function
        self.cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y,logits = self.disturbed_logits))
        self.reg_loss = self.lmd * tf.nn.l2_loss(self.W)
        self.loss = self.cross_entropy_loss + self.reg_loss
        
        ## compute accuracy
        correct_prediction = tf.equal(tf.argmax(self.disturbed_logits,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
       
        ## optimizer
        global_steps = tf.Variable(0, trainable=False)
        initial_learning_rate = self.lr
        steps_per_epoch = int(len(mnist.train.images)/ self.batch_size)
        decay_steps = 2 * steps_per_epoch
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_steps, decay_steps, self.decay, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss,var_list = [self.W],global_step=global_steps)  
        
        ## Training
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        init_fn(sess)
        
        ## restore if checkpoint flies exist
        ckpt = tf.train.get_checkpoint_state(self.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        total_batch = int(mnist.train.num_examples/self.batch_size)
        training_start = time()
        for epoch in range(self.max_epoch):
            epoch_start=time()
            for batch in range(total_batch):
                image_batch, label_batch = mnist.train.next_batch(self.batch_size)
                image_batch = np.reshape(image_batch, [-1, 28, 28, 1])
                _, train_loss, img_X_adv = sess.run([optimizer, self.loss, self.X_adv],\
                                         feed_dict = {input_images:image_batch, Y:label_batch})
                if batch % 5 == 0:
                    print('epoch:{:03d}/{:03d}, batch: {:04d}/{}, loss: {:.4f} '.format(\
                          epoch, self.max_epoch,batch,total_batch ,train_loss))
                    #print('global_steps:{}'.format(sess.run(global_steps)))
                    #print('learning rate:{}'.format(sess.run(learning_rate)))
            if (epoch+1) % self.save_freq == 0:
                saver.save(sess, os.path.join(self.train_dir, 'model_{:06d}.ckpt'.format(epoch+1)))
                print('model_{:06d}.ckpt saved'.format(epoch+1)) 
                for j in range(5):
                    scipy.misc.toimage(img_X_adv[j]).save(os.path.join(self.sample_dir,'epoch_{:06d}_{}.jpg'.format((epoch+1),j))) 
            epoch_duration =time()-epoch_start
            print("Training this epoch takes:","{:.2f}".format(epoch_duration))
        training_duration = time()-training_start

        ## Test when training finished
        testing_start = time()
        test_images = mnist.test.images 
        test_images = np.reshape(test_images, [-1, 28, 28, 1])
        test_labels = mnist.test.labels

        test_total_batch = int(len(mnist.test.images)/self.batch_size)
        test_acc_sum = 0.0
        for i in range(test_total_batch):
            test_image_batch = test_images[i*self.batch_size:(i+1)*self.batch_size]
            test_label_batch = test_labels[i*self.batch_size:(i+1)*self.batch_size]
            test_batch_acc = sess.run(accuracy, feed_dict = {input_images:test_image_batch,Y:test_label_batch})
            test_acc_sum += test_batch_acc
        test_acc = float(test_acc_sum/test_total_batch)
        testing_duration = time()-testing_start
        print('test_acc:{:.4f}'.format(test_acc)) 
        print("Training {:03d}".format(self.max_epoch)+" epoches takes:{:.2f} secs".format(training_duration))    
        print("Testing finished takes:{:.2f} secs".format(testing_duration))    

        coord.request_stop()
        coord.join(threads)
             
            
            
            
            

    
            
            
            
            
        

                                                                
