import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.models import resnet18
import os
# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels for ResNet18
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoader
batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define ResNet18 model
model = resnet18(weights=None, num_classes=10)  # 10 classes for MNIST
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 指定保存和加载的文件名
model_params_file = '/data/ResNetModel/ResNet_model_params.pth'

# 检查是否存在已保存的模型参数文件
if os.path.exists(model_params_file):
    # 如果存在，加载模型参数
    model.load_state_dict(torch.load(model_params_file))
    print("已加载保存的模型参数")
else:
    print("未找到模型参数，重新开始训练")
    # Training loop
    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        i=0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            i=i+1
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), model_params_file)
# Evaluate the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    i = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        i+=1
        print(f' Step [{i}/{len(test_loader)}]')

accuracy = correct / total
print(f"Accuracy on the test set: {accuracy:.4f}")

# Now, let's generate adversarial examples using FGSM
def fgsm_attack(image, epsilon, data_grad):
    perturbed_image = image + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def test_fgsm_attack(model, test_loader, epsilon):
    model.eval()
    correct = 0
    total = 0
    i=0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        data_grad = images.grad.data
        perturbed_images = fgsm_attack(images, epsilon, data_grad)

        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        i +=1
        print(f' Step [{i}/{len(test_loader)}]')

    accuracy = correct / total
    print(f"Accuracy on the adversarial examples (epsilon={epsilon}): {accuracy:.4f}")

# Test the model's robustness against FGSM attacks
epsilon = 0.0
test_fgsm_attack(model, test_loader, epsilon)

epsilon = 0.1
test_fgsm_attack(model, test_loader, epsilon)

epsilon = 0.2
test_fgsm_attack(model, test_loader, epsilon)

epsilon = 0.3
test_fgsm_attack(model, test_loader, epsilon)