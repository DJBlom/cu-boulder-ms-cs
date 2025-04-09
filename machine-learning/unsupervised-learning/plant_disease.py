##################################################################
# Content: CSCA 5632 Unsupervised Machine Learning Final Project
##################################################################
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision import models

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF

# --------------------- Utility Functions ---------------------
def cluster_accuracy(y_true, y_pred):
    """
    Compute clustering accuracy by matching predicted cluster labels with true labels.
    Uses the Hungarian algorithm to maximize the total number of correct predictions.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

def show_cluster_samples(dataset, cluster_labels, num_clusters, num_samples=3):
    """
    Save a few sample images from each cluster as separate PNG files.
    Each cluster's samples are saved as figure_sample_cluster_<cluster>.png.
    """
    for cl in range(num_clusters):
        indices = [i for i, label in enumerate(cluster_labels) if label == cl]
        if len(indices) == 0:
            continue
        chosen = random.sample(indices, min(num_samples, len(indices)))
        plt.figure(figsize=(12, 4))
        for i, idx in enumerate(chosen):
            image, _ = dataset[idx]
            # Convert image from CxHxW to HxWxC and denormalize
            image = image.permute(1, 2, 0).numpy()
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image = np.clip(image, 0, 1)
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(image)
            plt.title(f"Cluster {cl}")
            plt.axis('off')
        plt.suptitle(f"Samples from Cluster {cl}")
        filename = f"figure_sample_cluster_{cl}.png"
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

# --------------------- Custom Model ---------------------
class FineTuneModel(nn.Module):
    def __init__(self, num_clusters):
        super(FineTuneModel, self).__init__()
        # Initialize ResNet50 with the new 'weights' parameter to avoid deprecation warnings.
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        # Replace the final classification layer (not used for feature extraction)
        self.backbone.fc = nn.Linear(in_features, num_clusters)

    def forward(self, x):
        return self.backbone(x)

    def get_features(self, x):
        # Extract features from the backbone up to the avgpool layer.
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

# --------------------- Configuration ---------------------
# Update the path below to your dataset directory (assumes a folder structure: train/<class_name>/image.jpg)
data_dir = os.path.join("dataset", "train")  # <-- Update this path!

# Standard transformation: resize, tensor conversion, and normalization (using ImageNet stats)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load the dataset using ImageFolder (the folder names are only used for evaluation)
dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
true_labels = [label for _, label in dataset.samples]
num_classes = len(dataset.classes)  # This will also be used as the number of clusters

# Create DataLoader (no shuffling is needed for one-shot feature extraction)
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- Feature Extraction ---------------------
model = FineTuneModel(num_clusters=num_classes)
model = model.to(device)
model.eval()

features_list = []
with torch.no_grad():
    for inputs, _ in data_loader:
        inputs = inputs.to(device)
        feats = model.get_features(inputs)
        features_list.append(feats.cpu().numpy())
features_all = np.concatenate(features_list, axis=0)
print("Extracted features shape:", features_all.shape)

# --------------------- Apply NMF with max_iter=2000 ---------------------
# NMF requires non-negative input. Shift the features if needed.
min_val = features_all.min()
if min_val < 0:
    features_all_shifted = features_all - min_val + 1e-6
else:
    features_all_shifted = features_all.copy()

nmf = NMF(n_components=num_classes, init='nndsvda', random_state=32, max_iter=5000)
latent_features = nmf.fit_transform(features_all_shifted)
print("Latent features shape:", latent_features.shape)

# --------------------- Clustering with KMeans ---------------------
kmeans = KMeans(n_clusters=num_classes, random_state=32)
pseudo_labels = kmeans.fit_predict(latent_features)
final_acc = cluster_accuracy(true_labels, pseudo_labels)
print(f"Final Clustering Accuracy: {final_acc:.4f}")

# --------------------- EDA: Save Plots as PNG Files ---------------------
# Figure 1: Pseudo-Label Distribution Bar Chart
counts = np.bincount(pseudo_labels, minlength=num_classes)
plt.figure()
plt.bar(range(num_classes), counts)
plt.xlabel("Cluster Label")
plt.ylabel("Number of Images")
plt.title("Final Pseudo-Label Distribution")
plt.savefig("figure1.png")
print("Saved figure1.png")
plt.close()

# Figure 2: PCA Scatter Plot of Extracted Features (colored by pseudo-label)
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_all)
plt.figure(figsize=(10,8))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=pseudo_labels, cmap='viridis', alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA Projection of Features")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.savefig("figure2.png")
print("Saved figure2.png")
plt.close()

# Figure 3 and onward: Save sample images from each cluster
show_cluster_samples(dataset, pseudo_labels, num_clusters=num_classes, num_samples=3)
