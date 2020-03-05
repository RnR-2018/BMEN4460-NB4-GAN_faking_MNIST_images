# BMEN4460 Notebook 4.
## GAN faking MNIST images.
Nanyan "Rosalie" Zhu and Chen "Raphael" Liu

### Overview
This repository is a child repository of [**RnR-2018/Deep-learning-with-PyTorch-and-GCP**](https://github.com/RnR-2018/Deep-learning-with-PyTorch-and-GCP). This serves a primary purpose of facilitating the course BMEN4460 instructed by Dr. Andrew Laine and Dr. Jia Guo at Columbia University. However, it can also be used as a general beginner-level tutorial to implementing deep learning algorithms with PyTorch on Google Cloud Platform.

This repository, [**GAN faking MNIST images**](https://github.com/RnR-2018/RnR-2018-BMEN4460-NB4-GAN_faking_MNIST_images), presents an application of simple generative adverserial networks (GANs) on generating fake images.

For students in BMEN4460 (or who follow the instructions Step00 through Step02 in the parent repository), please create a Projects folder (if you have not done yet) within your GCP VM and download this repository into that folder.

On the GCP VM Terminal:
```
cd /home/[username]/
mkdir BMEN4460 # This is only necessary if you have not done this yet
mkdir BMEN4460/MNIST_GAN # This is only necessary if you have not done this yet
cd BMEN4460/MNIST_GAN
git clone https://github.com/RnR-2018/BMEN4460-NB2-image_classification_on_MNIST_data/
```

If it says "fatal: could not create work tree dir ...", you may as well try it again with super user permission
```
sudo git clone https://github.com/RnR-2018/BMEN4460-NB2-image_classification_on_MNIST_data/
```

You shall then see the following hierarchy of files and folders, hopefully, which matches the hierarchy of the current repository.

```
BMEN4460-NB4-GAN_faking_MNIST_images
    └── BMEN4460-NB4-GAN_faking_MNIST_images.ipynb
```

## Acknowledgements
This notebook is inspired by [this succinct GitHub repository](https://github.com/lyeoni/pytorch-mnist-GAN) and [this elaborate GitHub repository](https://github.com/Garima13a/MNIST_GAN), neither of which can be directly run/used due to issues with the PyTorch version of the MNIST dataset as well as some weird things with the training processes. Anyway we thank them for the resources, and we especially like the latter who walked through some nice concepts in detail.

Also, we later found [this beautiful GitHub repository](https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN) that is, no offense, way cooler than the other two. You are encouraged to take a look.
