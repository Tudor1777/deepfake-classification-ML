import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def load_data(csv_path, folder, bins=32):
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    for _, row in df.iterrows():
        image_path = os.path.join(folder, row[0] + ".png")
        image = Image.open(image_path).convert("RGB")
        histogram = extract_rgb_histogram(image, bins)
        images.append(histogram)
        labels.append(row[1])

    return np.array(images), np.array(labels)


def extract_rgb_histogram(image, bins=32):
    #creeaza vectorul de histograme pentru R, G, si B si le concateneaza
    r, g, b = image.split()
    rhist = np.histogram(np.array(r).flatten(), bins=bins, range=(0, 256))[0]
    ghist = np.histogram(np.array(g).flatten(), bins=bins, range=(0, 256))[0]
    bhist = np.histogram(np.array(b).flatten(), bins=bins, range=(0, 256))[0]
    return np.concatenate([rhist, ghist, bhist])

#incarca datele
training_images, training_labels = load_data("train.csv", "train")
validation_images, validation_labels = load_data("validation.csv", "validation")

# normalizeaza histogramele
scaler = StandardScaler()
training_images = scaler.fit_transform(training_images)
validation_images = scaler.transform(validation_images)

# antreneaza
svm = LinearSVC(max_iter=10000)
svm.fit(training_images, training_labels)

# evalueaza pe setul de validare
predictions = svm.predict(validation_images)
accuracy = np.mean(predictions == validation_labels)
print(f"Acuratete: {accuracy:.2%}")

matrix = confusion_matrix(validation_labels, predictions)
display = ConfusionMatrixDisplay(confusion_matrix=matrix)
display.plot()
plt.title("Matrice de confuzie")
plt.savefig("confusion_matrix_svm.png")
plt.show()
