# MNIST Handwritten Digit Recognition Project

Welcome to the **MNIST Handwritten Digit Recognition** project! This end-to-end machine learning repository demonstrates how to train, evaluate, and deploy a high-accuracy digit recognition model using TensorFlow/Keras, with both command-line and Streamlit-based web app interfaces.

## 📝 Project Overview

- **Purpose:**  
  Create an end-to-end workflow that classifies handwritten digits (0–9) from images, using the classic MNIST dataset.
- **Accuracy:**  
  Achieves **over 98% accuracy** on the official MNIST test set using a simple Convolutional Neural Network (CNN).
- **What you get:**  
  - Complete scripts for data loading, training, testing, and prediction
  - Test samples and utility scripts
  - An interactive Streamlit app for real-time digit drawing and prediction

## 📂 Repository Contents

| File/Folder               | Purpose                                                      |
|---------------------------|--------------------------------------------------------------|
| `train_mnist.py`          | Trains the CNN model on MNIST and saves the model file       |
| `mnist_loader.py`         | Utilities for loading raw MNIST `.idx` files                 |
| `test.py`                 | Script to test predictions using the Flask API or model      |
| `app.py`                  | Flask backend to serve prediction API and/or web interface   |
| `mnist_streamlit_app.py`  | Streamlit app for drawing/uploading digits and predicting    |
| `mnist_cnn_model.h5` / `.keras` | Saved trained model with 98%+ accuracy             |
| `templates/index.html`    | Frontend HTML for the Flask web app                          |
| `archive/`                | Extracted MNIST dataset files (`.idx` format)                |
