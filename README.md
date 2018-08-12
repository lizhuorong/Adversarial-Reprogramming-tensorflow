# Adversarial Reprogramming of Neural Networks
TensorFlow implementation of Adversarial Reprogramming of Neural Networks https://arxiv.org/abs/1806.11146

## Setup

### Prerequisites
- Python 2.7 or 3.x 
- TensorFlow 1.4 or higher
([install instructions](https://www.tensorflow.org/install/))
### Getting Started
- Clone this repo:
```bash
git clone git@github.com:lizhuorong/Adversarial-Reprogramming-tensorflow.git
cd Adversarial-Reprogramming-tensorflow
```
### Download the imagenet models
Download the following pre-trained models and put them into './model':
- [resnet_v2_50](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz)
- [inception_v3](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)
- [vgg_16](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)
- More imagenet models can be found [here](https://github.com/tensorflow/models/tree/master/research/slim)

### Datasets
- MNIST dataset will be automatically downloaded after running the scripts. 
- CIFAR-10. Training on CIFAR-10 have not been implemented yet. However, it is easy to adapt to more datasets and imagenet models.
## Train
Simply run the following command:
```
 python main.py 
```
You can train adversarial images for other ImageNet classifiers as well. <br>
For example, if you want to adversarially reprogram Inception-ResNet-v2, first you need to insert `from nets import inception_resnet_v2` in `model.py`. <br>
Then run:
```
 python main.py --network_name inception_resnet_v2
```
- More available networks can be found in the subfolder `./nets`, which is derived from [slim](https://github.com/tensorflow/models/tree/master/research/slim).
- Checkpoint files will be saved in `./train` and you are able to continue training your model from the previous epoch. 
- Sampled images can be found in `./sample`. Following are the examples that repurposing the ImageNet classifiers to MNIST classficaion.  ( top: Inception_V3 ; bottom : Resnet_v2_50)

<img src="imgs/concat.jpg" width="600px"/>

## Test
Test will be performed immediately after training finished.

## Results
The performance of adversarially reprogramming the trained ImageNet classifiers to perform MNIST classification.
Table gives test accuracy of reprogrammed networks on an MNIST classification task.

| ImageNet Model | MNIST |
|------|-------|
|Resnet_v2_50| 0.9586 |
|Inception_v3| 0.9745 |

## Acknowledgments
Code referred to a [Pytorch implementation](https://github.com/Prinsphield/Adversarial_Reprogramming). 
