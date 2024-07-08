# Image Generation with GANs

## Project Overview
This repository contains a project that uses Generative Adversarial Networks (GANs) to generate new images from existing datasets. The project applies GANs to Fashion MNIST and Abstract Art datasets, focusing on Deep Convolutional GANs (DCGANs) and other GAN variations. The methodology, implementation, and evaluation of the results are thoroughly documented.

## Table of Contents
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Sourcing Data](#sourcing-data)
- [Methodology](#methodology)
  - [GANs](#gans)
  - [Focus - DCGANs](#focus---dcgans)
  - [Other Types of GANs](#other-types-of-gans)
- [Math Behind GANs](#math-behind-gans)
- [Algorithms](#algorithms)
  - [Generator and Discriminator Code Blocks](#generator-and-discriminator-code-blocks)
  - [Training](#training)
- [Implementation](#implementation)
  - [Training Data Processing](#training-data-processing)
  - [Python in Google Colab](#python-in-google-colab)
  - [Challenges and Limitations](#challenges-and-limitations)
- [Evaluation and Final Results](#evaluation-and-final-results)
  - [Fashion Dataset](#fashion-dataset)
  - [Abstract Art](#abstract-art)
- [Contribution Table](#contribution-table)
- [References](#references)

## Introduction
Machine Learning algorithms can interpret images similarly to the human brain, aiding in various applications such as medical imaging, security, and object detection. This project explores using GANs for generating new images based on existing designs while retaining the patterns of the input datasets.

## Motivation
The project aims to apply the methodologies learned in class to a real-world problem, exploring the capabilities of GANs for image generation. The use of GANs in generating new designs from existing datasets is a compelling application of machine learning.

## Sourcing Data
We used two datasets for this project:
1. **Abstract Art**: Sourced from Kaggle, containing 2,782 high-resolution images of abstract art paintings.
2. **Fashion MNIST**: A dataset of Zalando’s article images, with 70,000 28x28 grayscale images across ten categories.

## Methodology
### GANs
GANs involve two competing neural networks: a generator that creates synthetic images and a discriminator that evaluates their authenticity. The generator improves over time to produce images indistinguishable from real ones.

### Focus - DCGANs
Deep Convolutional GANs (DCGANs) use convolutional and fractionally-strided convolutional layers to generate high-quality images. DCGANs are optimized for faster convergence and better image quality.

### Other Types of GANs
- **Conditional GAN (CGAN)**: Adds extra information to the generator and discriminator.
- **CycleGAN**: Translates images from one domain to another.
- **Generative Adversarial Text to Image Synthesis**: Generates images based on text descriptions.
- **Style GAN**: Focuses on improving the generator using Adaptive Instance Normalization.
- **Super Resolution GAN (SRGAN)**: Enhances the resolution of images.

## Math Behind GANs
GANs aim to minimize the error between the real and generated data distributions. The loss functions for the discriminator and generator guide the training process, with the ultimate goal of making the generated images indistinguishable from real images.

## Algorithms
### Generator and Discriminator Code Blocks
The generator and discriminator are built using PyTorch, employing layers such as Conv2D, ConvTranspose2D, BatchNorm2D, ReLU, and LeakyReLU.

### Training
The training process involves alternating updates to the discriminator and generator using stochastic gradient descent, with the goal of minimizing the adversarial loss.

## Implementation
### Training Data Processing
Training data is processed using transformations such as resizing, cropping, flipping, and normalization. DataLoader is used to create shuffled batches for training.

### Python in Google Colab
The project is implemented using PyTorch and Google Colab, leveraging GPU acceleration for faster training. Colab's features facilitate collaborative development and efficient execution.

### Challenges and Limitations
Challenges include hardware limitations and the complexity of GANs, which require extensive training and tuning to produce high-quality images. Mode collapse and training instability are common issues encountered during implementation.

## Evaluation and Final Results
### Fashion Dataset
The model was trained on the Fashion MNIST dataset, showing significant improvements in generated image quality over 50 epochs.

### Abstract Art
The model was also applied to the Abstract Art dataset, achieving impressive results in generating images reminiscent of the training data over 200 epochs.

## Contribution Table
| Task                | Contributor      |
|---------------------|------------------|
| Data Sourcing       | Oojas Salunke    |
| Model Implementation| Yash Bhole       |
| Training & Evaluation | Oojas Salunke  |
| Report Documentation | Yash Bhole      |

## References
1. Deep Learning by Python, book by François Chollet
2. GANs in Action, book by Jakub Langr
3. Opengenus Types of Generative Adversarial Networks (GANs)
4. The Math Behind GANs - JakeTae, Youtube video by Normalized Nerd
5. Data Science Stack Exchange
6. Medium: DCGAN: Deep Convolutional Generative Adversarial Network
7. Google Developers: Common Problems with GANs
8. YouTube videos: Edureka, IBM Tech, and Generating Pokemon with a GAN
9. Python Documentations: PyTorch, TensorFlow, and Keras
10. Machine Learning Mastery: GANs Training Tips
11. DCGAN Tutorial by PyTorch
