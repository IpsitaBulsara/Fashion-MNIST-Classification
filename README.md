
# 👚 Fashion MNIST Classification with PyTorch

This repository explores how well different deep learning models can classify images from the **Fashion MNIST** dataset — a set of 28x28 grayscale images of clothes like shirts, sneakers, and bags.

I’ve built and compared **three different models using PyTorch**:

## 🧠 Models Included

### 1. **ANN with Bayesian Hyperparameter Tuning**
A basic artificial neural network (ANN), where I used **Bayesian Optimization** to fine-tune hyperparameters like learning rate, number of layers, dropout rate, batch size, and number of neurons per layer.  
Helps find a better model setup with fewer trials.

### 2. **Custom CNN**
A **Convolutional Neural Network** built from scratch.  
Uses layers like `Conv2d`, `ReLU`, and `MaxPool2d` to capture important patterns in the images, improving performance over a simple ANN.

### 3. **VGG16 with Transfer Learning**
This one uses the **pretrained VGG16 model** from `torchvision.models`.  
I modified the classifier at the end to fit our 10 clothing categories.  
Since VGG16 was trained on millions of images from ImageNet, it already has strong feature extraction capabilities — and it shows.

## 🏆 Which One Performed Best?

**VGG16** came out on top!  
Thanks to its pretrained weights and deep architecture, it performed significantly better than the other models on Fashion MNIST.

| Model        | Description                      | Performance |
|--------------|----------------------------------|-------------|
| ANN          | With Bayesian tuning             | ⭐⭐☆☆☆       |
| CNN          | Custom architecture              | ⭐⭐⭐☆☆       |
| VGG16        | Pretrained + modified classifier | ⭐⭐⭐⭐⭐       |

## 📁 Files in This Repo

- `ann_bayes_tuning.ipynb` – ANN with hyperparameter tuning  
- `cnn_model.ipynb` – CNN model built from scratch  
- `vgg16_transfer_learning.ipynb` – VGG16 with updated classifier  


## 📦 About the Dataset

- Fashion MNIST is available directly from [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)  
- It includes 60,000 training images and 10,000 test images  
- Each image belongs to one of 10 clothing categories

