import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import numpy as np
import csv

# aici am incercat sa implementez functiile folosite intr-un cnn


def load_image(path):
    image = Image.open(path).convert("RGB")
    image_array = np.array(image)
    return image_array / 255.0

def load_data(path, image_folder):
    dataset = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            image_id, label = row[0], int(row[1])
            image_path = os.path.join(image_folder, image_id + ".png")
            if os.path.exists(image_path):
                image = load_image(image_path)
                dataset.append((image, label))
    return dataset



def convolve(image, filter, px):
    height, width = image.shape
    fh, fw = filter.shape
    feature_map = np.zeros((((height - fh) // px) + 1, ((width - fw) // px) + 1))
    for i in range(feature_map.shape[0]):
        for j in range(feature_map.shape[1]):
            region = image[i * px: i * px + fh, j * px: j * px + fw]
            feature_map[i, j] = np.sum(region * filter)
    return feature_map

def convolve_RGB(image, filters, px):
    n, _, fh, fw = filters.shape
    H, W, _ = image.shape
    feature_maps = np.zeros((n, (H - fh) // px + 1, (W - fw) // px + 1))
    for f in range(n):
        for i in range(feature_maps.shape[1]):
            for j in range(feature_maps.shape[2]):
                region = image[i * px:i * px + fh, j * px:j * px + fw, :]
                feature_maps[f, i, j] = np.sum(region * filters[f])
    return feature_maps

def relu(x):
    return np.maximum(0, x)

def max_pool(feature_map, size, px):
    height, width = feature_map.shape
    mpH = (height - size) // px + 1
    mpW = (width - size) // px + 1
    mp = np.zeros((mpH, mpW))
    for i in range(mpH):
        for j in range(mpW):
            window = feature_map[i * px:i * px + size, j * px:j * px + size]
            mp[i, j] = np.max(window)
    return mp

def max_pool_all(feature_maps, size, px):
    n, height, width = feature_maps.shape
    mpH = (height - size) // px + 1
    mpW = (width - size) // px + 1
    mp = np.zeros((n, mpH, mpW))
    for i in range(n):
        mp[i] = max_pool(feature_maps[i], size, px)
    return mp

def dense_layer(v, weights, bias):
    return np.dot(weights, v) + bias

def softmax(scores):
    exps = np.exp(scores - np.max(scores))
    return exps / np.sum(exps)

def cross_entropy_loss(prob, true_label):
    return -np.log(prob[true_label] + 1e-10)

def predict(image, filters, weights, bias):
    feature_maps = convolve_RGB(image, filters, px=1)
    activated = relu(feature_maps)
    pooled = max_pool_all(activated, size=2, px=2)
    flat = pooled.flatten()
    scores = dense_layer(flat, weights, bias)
    probs = softmax(scores)
    return probs, flat

def train_step(image, true_label, filters, weights, bias, lr):
    probs, flat = predict(image, filters, weights, bias)
    loss = cross_entropy_loss(probs, true_label)
    derivatives = probs.copy()
    derivatives[true_label] -= 1
    weights -= lr * np.outer(derivatives, flat)
    bias -= lr * derivatives
    return probs, loss


def train(train_dataset, val_dataset, filters, weights, bias, lr, nr_epochs):
    for epoch in range(nr_epochs):
        total_loss = 0
        for image, label in train_dataset:
            _, loss = train_step(image, label, filters, weights, bias, lr)
            total_loss += loss
        average_loss = total_loss / len(train_dataset)
        print(f"Epoca {epoch+1}, Pierdere: {average_loss:.4f}")

        # evaluare pe setul de evaluare
        correct = 0
        total = 0
        for image, label in val_dataset:
            probs, _ = predict(image, filters, weights, bias)
            prediction = np.argmax(probs)
            correct += (prediction == label)
            total += 1
        accuracy = correct / total
        print(f"Validare: {correct}/{total} = {accuracy:.2%}")

        with open("manual_train_log.txt", "a") as f:
            f.write(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}\n")



train_data = load_data("train.csv", "train")
validation_data = load_data("validation.csv", "validation")
filters = np.random.randn(8, 3, 3, 3) * 0.01

# aici gasesc dimensiunea vectorului care intra in fully connected layer
image, _ = train_data[0]
feature_maps = convolve_RGB(image, filters, px=1)
activated = relu(feature_maps)
pooled = max_pool_all(activated, size=2, px=2)
flat_dim = pooled.flatten().shape[0]

weights = np.random.randn(5, flat_dim) * 0.01
bias = np.zeros(5)

train(train_data, validation_data, filters, weights, bias, lr=0.01, n_epochs=30)


#aici am vrut sa vad cat dureaza sa treaca prin 100 de imagini(aproape un minut)
start = time.time()
for image, label in train_data[:100]:
    predict(image, filters, weights, bias)
print("Timp pentru 100 imagini:", time.time() - start)