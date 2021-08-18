# FS-DGPM

This repository is the official implementation of "[Flattening Sharpness for Dynamic Gradient Projection Memory Benefits Continual Learning](https://arxiv.org/pdf/)".

## Abstract

The backpropagation networks are notably susceptible to catastrophic forgetting, where networks tend to forget previously learned skills upon learning new ones. To address such the 'sensitivity-stability' dilemma, most previous efforts have been contributed to minimizing the empirical risk with different parameter regularization terms and episodic memory, but rarely exploring the usages of the weight loss landscape. In this paper, we investigate the relationship between the weight loss landscape and sensitivity-stability in the continual learning scenario, based on which, we propose a novel method, Flattening Sharpness for Dynamic Gradient Projection Memory (FS-DGPM). In particular, we introduce a soft weight to represent the importance of each basis representing past tasks in GPM, which can be adaptively learned during the learning process, so that less important bases can be dynamically released to improve the sensitivity of new skill learning. We further introduce Flattening Sharpness (FS) to reduce the generalization gap by explicitly regulating the flatness of the weight loss landscape of all seen tasks. As demonstrated empirically, our proposed method consistently outperforms baselines with the superior ability to learn new skills while alleviating forgetting effectively.

## Requisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.9.5
- torch = 1.9.0
- torchvision = 0.10.0

To get started, please install the requirements inside your environment using conda. Type the following in your terminal:

```conda env create -f environment.yml```

Once completed source your environment using:

```conda activate fsdgpm```

## Available Datasets

The code works for Permuted MNIST (PMNIST), CIFAR100 Split, CIFAR100 Superclass, and TinyImageNet. 

CIFAR100 Split and Superclass is automatically downloaded when you run a script for CIFAR experiments.

For PMNIST and TinyImageNet, run the following commands: 

```cd data```

```python get_data.py```

```source download_tinyimgnet.sh```


## How to use it

In run_experiments.sh see examples of how to run FS-DGPM for Permuted MNIST, 10-split CIFAR-100, 20-tasks CIFAR-100 Superclass and TinyImageNet. All these experiments can be run using the following command:

```
source run_experiments.sh
```


## Citation

```

```
