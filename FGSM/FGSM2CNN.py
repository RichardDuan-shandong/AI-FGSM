import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms  # Add this line
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=60000, shuffle=True)

# Extract images and labels from the training set
for images, labels in train_loader:
    break

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(14 * 14 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load the pre-trained model
model = SimpleCNN()
model.load_state_dict(torch.load('/data/CNNModel/cnn_model_params.pth'))
model.eval()

## Extract features from the model
def extract_features(model, data_loader):
    features = []
    labels = []
    for images, labels_batch in data_loader:
        with torch.no_grad():
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(labels_batch.cpu().numpy())
    features = np.vstack(features)
    labels = np.concatenate(labels)
    return features, labels

# Extract features and labels from the test set
features, labels = extract_features(model, train_loader)

# Use PCA for dimensionality reduction to 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)

# Standardize the data
scaler = StandardScaler()
pca_result = scaler.fit_transform(pca_result)

# Plot decision boundaries
plt.figure(figsize=(8, 8))
for i in range(10):
    if i==8:
        i=9
    else:
        if i==9:
            i=8
    indices = labels == i
    plt.scatter(pca_result[indices, 0], pca_result[indices, 1], label=str(i), alpha=0.5)

plt.title('Decision Boundaries of the CNN Model on MNIST (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
