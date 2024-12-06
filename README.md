# MNIST_CNN_tuning
Identify best light model for MNIST accuracy.


## Overview
This repository contains code for training and evaluating CNN models on the MNIST dataset, with a focus on finding lightweight architectures that maintain good accuracy.

## Files
- `Train.py`: Main training script that:
  - Loads and preprocesses MNIST dataset
  - Defines and trains CNN model architectures
  - Saves trained models and training metrics
  - Supports hyperparameter tuning

- `Test.py`: Testing and evaluation script that:
  - Loads trained models
  - Runs inference on test dataset
  - Calculates accuracy and performance metrics
  - Includes test cases for:
    - Total Parameter Count
    - Use of Batch Normalization 
    - Use of DropOut
    - Use of Fully Connected Layer or GAP
   



## Usage
1. Train a model

## Requirements
- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- tqdm

## Training logs for 20 epochs

loss=0.12099608033895493 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.32it/s]  <br />
Test set: Average loss: 0.1098, Accuracy: 9698/10000 (97%)

loss=0.045415740460157394 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.37it/s] <br />
Test set: Average loss: 0.0563, Accuracy: 9834/10000 (98%)

loss=0.09385127574205399 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.20it/s] <br />
Test set: Average loss: 0.0444, Accuracy: 9858/10000 (99%)

loss=0.08695608377456665 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.88it/s] <br />
Test set: Average loss: 0.0327, Accuracy: 9899/10000 (99%)

loss=0.025562001392245293 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.36it/s] <br />
Test set: Average loss: 0.0347, Accuracy: 9892/10000 (99%)

loss=0.060603633522987366 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 24.98it/s] <br />
Test set: Average loss: 0.0297, Accuracy: 9894/10000 (99%)

loss=0.07441337406635284 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.22it/s] <br />
Test set: Average loss: 0.0274, Accuracy: 9908/10000 (99%)

loss=0.033080533146858215 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.96it/s] <br />
Test set: Average loss: 0.0249, Accuracy: 9908/10000 (99%)

loss=0.03273075073957443 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.01it/s] <br />
Test set: Average loss: 0.0249, Accuracy: 9917/10000 (99%)

loss=0.02988106943666935 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 24.84it/s] <br />
Test set: Average loss: 0.0290, Accuracy: 9911/10000 (99%)

loss=0.010840273462235928 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.74it/s] <br />
Test set: Average loss: 0.0236, Accuracy: 9926/10000 (99%)

loss=0.020415594801306725 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 24.78it/s] <br />
Test set: Average loss: 0.0212, Accuracy: 9928/10000 (99%)

loss=0.09163998812437057 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 24.75it/s] <br />
Test set: Average loss: 0.0185, Accuracy: 9943/10000 (99%)

loss=0.06761204451322556 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.47it/s] <br />
Test set: Average loss: 0.0232, Accuracy: 9919/10000 (99%)

loss=0.08394000679254532 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.66it/s] <br />
Test set: Average loss: 0.0225, Accuracy: 9926/10000 (99%)

loss=0.03609871864318848 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.20it/s] <br />
Test set: Average loss: 0.0204, Accuracy: 9933/10000 (99%)

loss=0.02191767655313015 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.94it/s] <br />
Test set: Average loss: 0.0203, Accuracy: 9938/10000 (99%)

loss=0.026561526581645012 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.99it/s] <br />
Test set: Average loss: 0.0178, Accuracy: 9941/10000 (99%)

loss=0.011089473962783813 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.21it/s] <br />
Test set: Average loss: 0.0174, Accuracy: 9944/10000 (99%)

loss=0.028131581842899323 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.07it/s] <br />
Test set: Average loss: 0.0179, Accuracy: 9944/10000 (99%)
