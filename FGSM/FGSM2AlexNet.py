import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
# Define the AlexNet model
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.MNIST(root='/data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='/data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=180, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=180, shuffle=False)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 指定保存和加载的文件名
model_params_file = '/data/AlexNetModel/AlexNet_model_params.pth'

# 检查是否存在已保存的模型参数文件
if os.path.exists(model_params_file):
    # 如果存在，加载模型参数
    model.load_state_dict(torch.load(model_params_file))
    print("已加载保存的模型参数")
else:
    print("未找到模型参数，重新开始训练")
    # Train the model
    num_epochs = 3

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), model_params_file)



# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on the test set: {100 * correct / total}%')

# FGSM attack function
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Clip the perturbed image to maintain pixel values in the valid range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Attack on a test sample
model.eval()
image, label = test_dataset[0]
image = image.unsqueeze(0).to(device)
label = torch.tensor([label]).to(device)

# Enable gradient calculation for the input image
image.requires_grad = True

output = model(image)
loss = criterion(output, label)
model.zero_grad()
loss.backward()

# Collect the gradient of the input image
data_grad = image.grad.data

# Set the epsilon value for the FGSM attack
epsilon = 0.1

# Generate adversarial example using FGSM
perturbed_image = fgsm_attack(image, epsilon, data_grad)

# Test the model on the adversarial example
model.eval()
with torch.no_grad():
    adversarial_output = model(perturbed_image)
    _, predicted = torch.max(adversarial_output.data, 1)

print(f'Original label: {label.item()}, Predicted label on adversarial example: {predicted.item()}')