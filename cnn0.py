import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd


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



# mosteneste nn.Module din PyTorch
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #straturi convolutionale
        self.convolve1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.convolve2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.convolve3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.dense_layer = nn.Linear(128 * 12 * 12, 5)  # un singur fully-connected layer, am 128 feature maps de imagini 12x12

    def forward(self, x):
        #cele 3 straturi convolutionale
        x = self.max_pool(F.relu(self.convolve1(x)))
        x = self.max_pool(F.relu(self.convolve2(x)))
        x = self.pool(F.relu(self.convolve3(x)))
        x = self.flatten(x)
        #singurul strat fully connected
        return self.dense_layer(x)


#normalizare
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

#incarc datele
train_data = Data("train.csv", "train", transform)
validation_data = Data("validation.csv", "validation", transform)

training = DataLoader(train_data, batch_size=32, shuffle=True)
validation = DataLoader(validation_data, batch_size=32)


#initializeaza modelul
model = CNN()
#initializeaza functia de pierdere si optimizatorul
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#antrenare pt 30 de epoci
for epoch in range(30):
    model.train()
    running_loss = 0.0
    #aici itereaza prin setul de antrenare
    for images, labels in training:
        optimizer.zero_grad() #reseteaza gradientele din ultima epoca
        outputs = model(images) #apeleaza forward si calculeaza predictiile
        loss = criterion(outputs, labels) #calculeaza pierderea
        loss.backward() #calculeaza gradientele
        optimizer.step() #actualizeaza parametrii pe baza gradientelor
        running_loss += loss.item()
    #pierderea pt epoca curenta
    epoch_loss = running_loss / len(training)
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    #evaluare pe setul de validare
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in validation:
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1) #selecteaza clasa cu cea mai mare probabilitate
            correct += (predictions == labels).sum().item() #verifica daca e corect
            total += labels.size(0)
    #calculeaza acuratetea
    accuracy = correct / total
    print(f"Validare: {correct}/{total} = {accuracy:.2%}")
