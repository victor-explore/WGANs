# WGAN with DiffAugment for Animal Image Generation

This project implements a Wasserstein Generative Adversarial Network (WGAN) with Differential Augmentation (DiffAugment) for generating animal images.

## Overview

The notebook `Jupyter notebook.ipynb` contains the implementation of a WGAN model trained on an animal image dataset. It utilizes DiffAugment to improve training stability and image quality, especially when working with limited data.

## Key Components

1. **Generator**: A convolutional neural network that generates images from random noise.
2. **Discriminator**: A convolutional neural network that distinguishes between real and generated images.
3. **DiffAugment**: A differentiable augmentation technique applied to both real and generated images during training.
4. **WGAN Loss**: Wasserstein loss for improved stability in GAN training.

## Features

- Custom `SimpleImageDataset` class for loading and preprocessing animal images
- WGAN architecture with gradient penalty
- DiffAugment for data augmentation
- Learning rate scheduling with ExponentialLR
- Model checkpointing and image saving during training

## Requirements

- PyTorch
- torchvision
- matplotlib
- numpy
- PIL
- tqdm

## Usage

1. Set up the required environment and dependencies.
2. Prepare your animal image dataset.
3. Update the data directory path in the notebook.
4. Run the cells in the Jupyter notebook sequentially.

## Training Process

The training loop includes:
- Loading and augmenting real images
- Generating fake images
- Training the discriminator and generator alternately
- Applying gradient penalty for WGAN
- Saving generated images and model checkpoints

## Results

Generated images and model checkpoints are saved in specified directories throughout the training process.

## Note

This implementation is designed for research and educational purposes. The quality of generated images may vary depending on the dataset and hyperparameters used.
