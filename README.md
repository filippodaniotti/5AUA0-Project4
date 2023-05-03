# 5AUA0 Template

This template contains a minimal implementation of a convolutional neural network for image classification.

## Requirements
This code has been tested with the following versions:
- python == 3.10
- pytorch == 2.0
- torchvision == 0.15
- numpy == 1.23
- pillow == 9.4
- cudatoolkit == 11.7 (Only required for using GPU & CUDA)

We recommend you to install these dependencies using [Anaconda](https://docs.anaconda.com/anaconda/install/). With Anaconda installed, the dependencies can be installed with
```bash
conda create --name 5aua0 python=3.10
conda activate 5aua0
conda init
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```     

## Dataset
To download and prepare the dataset, follow these steps:
- Download the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) from [this URL](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).
- Unpack the file and store the `cifar-10-batches-py` directory in the `data` directory of this repository.
- To prepare the dataset, run `python prepare_cifar10_data.py`.

## Training and testing the network
To train the network, run:
```bash
python train.py
```
This will train the network with the configuration settings stored in `config.py`.

To test the network, run:
```bash
python test.py
```
This will test the network on the testing set and report the accuracy. After training and testing the network using the default configuration settings, an accuracy of approximately `64%` should be achieved.


## Config and hyperparameters
The hyperparameters are defined in `config.py`. We define:
- `batch_size_train`: The batch size during training (default: 32)
- `batch_size_test`: The batch size during testing (default: 25)
- `lr`: The learning rate (default: 0.005)
- `lr_momentum`: The learning rate momentum for the SGD optimizer (default: 0.9)
- `weight_decay`: The weight decay (default: 1e-4)
- `num_classes`: The number of classes to be classified (default: 10 for CIFAR10)
- `gt_dir`: Location where the dataset/ground-truth is stored (default: `"./data/cifar-10-batches-py/"`)
- `num_iterations`: The number of iterations to train the network (default: 10000)
- `log_iterations`: The number of iterations after which the loss is logged and reported (default: 100)
- `enable_cuda`: Enabling CUDA. NOTE: only possible if PyTorch is installed with CUDA, and if a GPU is available (default: False)


## Potential improvements
As mentioned, this is a very minimal example. This code should be changed to solve your task, and there are plenty of functionalities that you could add to make your life easier, or to improve the performance. Some examples:
- Store the `config` info for each training that you do, so you remember the hyperparameters you used.
- Run evaluation on the validation/test set after every X steps, to keep track of the performance of the network on unseen data.
- Try out other network components, such as `BatchNorm` and `Dropout`.
- Fix random seeds, to make results (and bugs) better reproducible.
- Visualize the results of your network! By visualizing the input image and printing/logging the predicted result, you can get insight in the performance of your model.
- And of course, improve network and the hyperparameters to get a better performance!

## HPC
In order to run this repository on the Snellius High Performance Computing platform, please follow our [HPC Tutorial](https://tue-5aua0.github.io/hpc_tutorial.html)