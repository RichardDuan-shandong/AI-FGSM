import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets, models
import os

class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()

        # 输入层
        self.flatten = nn.Flatten()

        # 第一个隐藏层
        self.fc1 = nn.Linear(28*28, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        # 第二个隐藏层
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        # 第三个隐藏层
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        # 输出层
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)

        # 第一个隐藏层的前向传播
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # 第二个隐藏层的前向传播
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # 第三个隐藏层的前向传播
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        # 输出层的前向传播
        x = self.fc4(x)

        return x
# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = datasets.MNIST("/data", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST("/data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
model = DNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 指定保存和加载的文件名
model_params_file = '/data/DNNModel/dnn_model_params.pth'

# 检查是否存在已保存的模型参数文件
if os.path.exists(model_params_file):
    # 如果存在，加载模型参数
    model.load_state_dict(torch.load(model_params_file))
    print("已加载保存的模型参数")
else:
    print("未找到模型参数，重新开始训练")
    # 训练模型
    num_epochs = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 训练DNN模型
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            dnn_outputs = model(images.view(images.size(0), -1))  # 注意将图像展平作为DNN的输入
            loss = criterion(dnn_outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"DNN_Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    torch.save(model.state_dict(), model_params_file)

# FGSM攻击
def fgsm_attack(image, epsilon, data_grad):

    # 攻击生成的对抗样本 = 原始图像 + epsilon * sign(梯度)
    disturbed_image = image + epsilon * data_grad.sign()
    # 将像素值裁剪到合法范围 [0, 1]
    disturbed__image = torch.clamp(disturbed_image, 0, 1)
    return disturbed_image

def test_fgsm(model, test_loader, epsilon):
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.requires_grad_(), labels
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # 获取输入图像的梯度
        data_grad = images.grad.data

        # FGSM攻击
        perturbed_images = fgsm_attack(images, epsilon, data_grad)

        # 在对抗样本上进行推理
        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"epsilon={epsilon}: {accuracy * 100:.2f}%")

# 测试模型在原始测试集上的性能
print("Accuracy(Original):")
test_fgsm(model, test_loader, epsilon=0.0)

# 测试模型在对抗样本上的性能
print("\nAccuracy(Adversarial):")
test_fgsm(model, test_loader, epsilon=0.07)

# 测试模型在对抗样本上的性能
print("\nAccuracy(Adversarial):")
test_fgsm(model, test_loader, epsilon=0.1)

# 测试模型在对抗样本上的性能
print("\nAccuracy(Adversarial):")
test_fgsm(model, test_loader, epsilon=0.12)

# 测试模型在对抗样本上的性能
print("\nAccuracy(Adversarial):")
test_fgsm(model, test_loader, epsilon=0.17)

# 测试模型在对抗样本上的性能
print("\nAccuracy(Adversarial):")
test_fgsm(model, test_loader, epsilon=0.2)

# 测试模型在对抗样本上的性能
print("\nAccuracy(Adversarial):")
test_fgsm(model, test_loader, epsilon=0.25)

# 测试模型在对抗样本上的性能
print("\nAccuracy(Adversarial):")
test_fgsm(model, test_loader, epsilon=0.3)

# 测试模型在对抗样本上的性能
print("\nAccuracy(Adversarial):")
test_fgsm(model, test_loader, epsilon=0.35)