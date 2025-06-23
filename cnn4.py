import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Data(Dataset):
    def __init__(self, csv, folder, transform=None):
        self.rows = pd.read_csv(csv)
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        image_name = self.rows.iloc[idx, 0] + ".png"
        label = int(self.rows.iloc[idx, 1])
        image_path = os.path.join(self.folder, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #am introdus functii de batch normalisation, pentru a stabiliza datele in timpul antrenarii
        self.convolve1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.convolve2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.convolve3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.dense_layer1 = nn.Linear(128 * 12 * 12, 256)
        self.dense_layer2 = nn.Linear(256, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.max_pool(F.relu(self.bn1(self.convolve1(x))))
        x = self.max_pool(F.relu(self.bn2(self.convolve2(x))))
        x = self.max_pool(F.relu(self.bn3(self.convolve3(x))))
        x = self.flatten(self.dropout(x))
        x = self.dense_layer1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.dense_layer2(x)


transform_training = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

transform_validation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_data = Data("train.csv", "train", transform=transform_training)
validation_data = Data("validation.csv", "validation", transform=transform_validation)

training = DataLoader(train_data, batch_size=32, shuffle=True)
validation = DataLoader(validation_data, batch_size=32)

model = CNN()
class_counts = [250, 250, 250, 250, 180]#vector de frecventa pe care l-am adaugat pentru a creste ponderea clasei 4
weights = torch.tensor([1 / c for c in class_counts])
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#daca acuratetea nu se imbunatateste dupa 3 epoci, se injumatateste rata de invatare, previne stagnarea
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

best_accuracy= 0
best_model = None
all_losses = []
all_accuracies = []


nr_epochs = 45
for epoch in range(nr_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in training:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(training)
    all_losses.append(epoch_loss)

    model.eval()
    correct = 0
    total = 0
    prediction_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in validation:
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            prediction_list.extend(predictions.tolist())
            labels_list.extend(labels.tolist())

    accuracy = correct / total
    all_accuracies.append(accuracy)
    print(f"Epoca {epoch + 1}, Pierdere: {epoch_loss:.4f}, Acuratete: {accuracy:.2%}\n")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model.state_dict()
        best_predictions = prediction_list
        best_labels = labels_list
    scheduler.step(accuracy)
torch.save(best_model, "best_model4.pth")

matrix = confusion_matrix(best_labels, best_predictions)
display = ConfusionMatrixDisplay(matrix)
display.plot()
plt.title("Matrice de confuzie")
plt.savefig("confusion_matrix4.png")

plt.figure()
plt.plot(range(1, nr_epochs + 1), all_losses, label="Loss", color='red')
plt.plot(range(1, nr_epochs + 1), all_accuracies, label="Accuracy", color='green')
plt.xlabel("Epoca")
plt.ylabel("Valoare")
plt.title("Evolutia pierderii si acuratetei")
plt.legend()
plt.savefig("training_curve4.png")