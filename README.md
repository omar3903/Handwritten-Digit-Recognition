# Handwritten Digit Recognition

Transfer learning with an ImageNet-pretrained ResNet18 in PyTorch to recognise my own handwritten digits. An individual image processing and recognition project from my Computer Science degree at Wilfrid Laurier University.

**Try it live:** [omar3903.dev](https://omar3903.dev/#digit-recognition). Draw a digit, or pick one of my test digits, and compare all three models running in your browser.

## The question

A model trained on MNIST does well on MNIST. How well does it read real handwriting it has never seen, and how much does fine-tuning on a small set of that handwriting help?

## Dataset

I wrote and photographed 120 digits, 12 of each digit 0–9. Each crop is named `<digit>_<n>.jpg`:

| Images | Use | Count |
|---|---|---|
| `_1` to `_8` | Training | 80 |
| `_9` and `_10` | Validation | 20 |
| `_11` and `_12` | Test | 20 |

The full photographed sheets are included too (`Training_Set_*.jpg`, `Validation_Set_*.jpg`, `Test_Set_*.jpg`).

## Method

All three models start from torchvision's `resnet18` with ImageNet weights:

- The first convolution is replaced by a 3×3, stride-1, single-channel layer for grayscale input. It is randomly initialised and frozen with the rest of the backbone.
- The final layer is replaced by a 10-class linear layer.
- Images are read as grayscale with OpenCV, resized to 28×28 and normalised with MNIST's mean and standard deviation.

| Model | Trained layers | Trained on | Settings |
|---|---|---|---|
| 1 | Final layer | MNIST (50,000 train / 10,000 validation) | Batch 128, SGD lr 0.01, momentum 0.9, StepLR (7, 0.1), 35 epochs |
| 2 | Final layer | My 80 training digits | Batch 16, SGD lr 0.001, momentum 0.9, 40 epochs |
| 3 | `layer4` + final layer | My 80 training digits | Batch 16, SGD lr 0.001, momentum 0.9, 40 epochs |

## Results

| Model | Training | Validation | Test |
|---|---|---|---|
| 1 on MNIST (10,000 test images) | 83.2% | 82.0% | **81.9%** |
| 1 on my handwriting | | | **15%** |
| 2 on my handwriting | 100% | 90% | **65%** |
| 3 on my handwriting | 100% | 95% | **80%** |

The MNIST-trained model reads only 15% of my digits, a clear case of domain shift. Training the final layer on my own photos lifts that to 65%, and also training the last residual block reaches 80%.

## What I found later

While building the live demo, I retrained all three models with this same code (they scored 10%, 75% and 85% on the test digits) and found a shortcut. My photos were taken under different lighting for different digits: bright paper for the 9s, shadow for the 7s and 8s. With only 80 training images, the models partly learned that brightness instead of the digits' shapes. Evening out each test photo's brightness drops the retrained Model 3 from 85% to 55%.

Two details of the code are worth knowing:

- `train_model` calls `model.train()` for every epoch, so the BatchNorm statistics of the frozen backbone are re-estimated on the training data. This turns out to drive much of the accuracy.
- The best-weights restore uses `model.state_dict().copy()`, which is a shallow copy, so the final-epoch weights are the ones used, not the best-validation ones.

The results sheet labels the model ResNet34, but the code and the report use ResNet18.

## Running it

```bash
pip install -r requirements.txt
```

`load_data()` reads the images from the working directory, and the MNIST files are read from `./MNIST/raw`. Copy every image from the three set folders into `Code/` (where `MNIST/raw` already is), then run:

```bash
cd Code
python Assignment4.py
```

It uses a CUDA GPU if one is available.

## Files

- `Code/Assignment4.py`: data loading, model setup, training and evaluation for all three models
- `Code/MNIST/raw/`: the MNIST dataset (LeCun, Cortes & Burges)
- `Training Set/`, `Validation Set/`, `Test Set/`: my handwritten digits
- `Assignment 4_Results.xlsx`: the results table
- `Assignment4_Report.pdf`: the brief report (student number removed)
- `requirements.txt`

The training loop is based on the [PyTorch transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
