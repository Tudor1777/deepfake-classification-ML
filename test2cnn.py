import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd

class TestData(Dataset):
    def __init__(self, csv, folder, transform=None):
        self.data = pd.read_csv(csv)
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image_name = self.data.iloc[i, 0]
        path = os.path.join(self.folder, image_name + ".png")
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_name


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolve1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.convolve2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.convolve3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.convolve4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.dense_layer1 = nn.Linear(128 * 12 * 12, 512)
        self.dense_layer2 = nn.Linear(512, 256)
        self.dense_layer3 = nn.Linear(256, 5)

    def forward(self, x):
        x = self.max_pool(F.relu(self.bn1(self.convolve1(x))))
        x = self.max_pool(F.relu(self.bn2(self.convolve2(x))))
        x = self.max_pool(F.relu(self.bn3(self.convolve3(x))))
        x = self.max_pool(F.relu(self.bn4(self.convolve4(x))))
        x = self.flatten(self.dropout(x))
        x = F.relu(self.dense_layer1(x))
        x = self.dropout(x)
        x = F.relu(self.dense_layer2(x))
        x = self.dropout(x)
        return self.dense_layer3(x)


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class TestData(Dataset):
    def __init__(self, csv, folder, transform=None):
        self.data = pd.read_csv(csv)
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx, 0]
        path = os.path.join(self.folder, img_id + ".png")
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_id

model = CNN()
model.load_state_dict(torch.load("best_model6.pth"))
model.eval()

test_data = TestData("test.csv", "test", transform=transform_test)
test_loader = DataLoader(test_data, batch_size=32)

results = []

with torch.no_grad():
    for images, ids in test_loader:
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        for image_id, pred in zip(ids, predictions):
            results.append((image_id, pred.item()))

out = pd.DataFrame(results, columns=["image_id", "label"])
out.to_csv("test_predictions6b.csv", index=False)
print("Predictiile au fost salvate")