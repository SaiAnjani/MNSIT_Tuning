# MNIST_CNN_tuning
Identify best light model for MNIST accuracy.


## Overview
This repository contains code for training and evaluating CNN models on the MNIST dataset, with a focus on finding lightweight architectures that maintain good accuracy.

## Files
- `Train.py`: Main training script that:
  - Loads and preprocesses MNIST dataset
  - Defines and trains CNN model architectures
  - Saves trained models and training metrics
  - Supports hyperparameter tuning via command line arguments

- `Test.py`: Testing and evaluation script that:
  - Loads trained models
  - Runs inference on test dataset
  - Calculates accuracy and performance metrics
  - Includes test cases for:
    - Model loading
    - Inference speed
    - Memory usage
    - Accuracy thresholds

## Usage
1. Train a model

## Requirements
- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- tqdm

