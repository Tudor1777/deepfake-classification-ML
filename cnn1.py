import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt


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
        self.convolve1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.convolve2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.convolve3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4) #am adaugat dropout ca sa previn suprainvatarea
        self.flatten = nn.Flatten()
        self.dense_layer1 = nn.Linear(128 * 12 * 12, 256)
        self.dense_layer2 = nn.Linear(256, 5) #inca un strat fully connected

    def forward(self, x):
        x = self.max_pool(F.relu(self.convolve1(x)))
        x = self.max_pool(F.relu(self.convolve2(x)))
        x = self.max_pool(F.relu(self.convolve3(x)))
        x = self.flatten(self.dropout(x))
        x = self.dense_layer1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.dense_layer2(x)


#am adaugat augmentari
transform_training = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

accuracies = []


for epoch in range(35):
    running_loss = 0.0
    for images, labels in training:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(training)
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")


    with open("train_log.txt", "a") as f:
        f.write(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}\n")


    torch.save(model.state_dict(), f"model_epoch{epoch + 31}.pth")
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in validation:
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Validare: {correct}/{total} = {accuracy:.2%}")
    accuracies.append(accuracy)

    # Log validare
    with open("train_log.txt", "a") as f:
        f.write(f"Epoca {epoch + 1}, Pierdere: {epoch_loss:.4f}, Acuratete: {accuracy:.2%}\n")


plt.plot(range(1, 36), accuracies, label="Accuracy")
plt.xlabel("Epoca")
plt.ylabel("Valori")
plt.legend()
plt.savefig("training_curve.png")