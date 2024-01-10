import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.autograd import Variable
import numpy as np
from sklearn.decomposition import PCA
import os
import cv2
# 定义简单的卷积神经网络模型
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

# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = datasets.MNIST("/data", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST("/data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 指定保存和加载的文件名
model_params_file = '/data/CNNModel/cnn_model_params.pth'

# 检查是否存在已保存的模型参数文件
if os.path.exists(model_params_file):
    # 如果存在，加载模型参数
    model.load_state_dict(torch.load(model_params_file))
    print("已加载保存的模型参数")
else:
    print("未找到模型参数，重新开始训练")
    # 训练模型
    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    torch.save(model.state_dict(), model_params_file)

def plt_show(sample_image,heatmap,result):
    # 可视化结果
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    sample_image = np.squeeze(sample_image)  # 去掉额外的维度
    ax1.imshow(sample_image, cmap='gray')  # 显示灰度图像
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Grad-CAM热力图
    ax2.imshow(heatmap)
    ax2.set_title('Grad-CAM Heatmap')
    ax2.axis('off')

    # 叠加结果
    result = np.squeeze(result)  # 去掉额外的维度
    ax3.imshow(result)  # 显示灰度图像
    ax3.set_title('Attention Image')
    ax3.axis('off')

    plt.show()

# 执行fgsm
def fgsm_t_attack(model, image, data_grad, epsilon):

    perturbation = epsilon * data_grad.sign()
    #show_image(perturbation[1])
    adversarial_image = image + perturbation
    adversarial_image = torch.max(torch.min(adversarial_image, image + epsilon), image - epsilon)
    adversarial_image = torch.clamp(adversarial_image, 0, 1)

    return adversarial_image

def GET_Grad(model, sample_image,target_class):
    # 前向传播获取预测结果
    output = model(sample_image)

    # 计算梯度
    model.zero_grad()
    output[0, target_class].backward()

    # 获取第1个卷积层的梯度
    gradients = model.conv1.weight.grad
    return gradients

def calculate_weights(gradients):

    # 计算CAM
    attention_weights = gradients.abs().sum(dim=0, keepdim=True)
    # 插值操作将CAM的大小调整为与输入图像相同
    attention_weights = nn.functional.interpolate(attention_weights, size=(28, 28), mode="bilinear",align_corners=False)
    # 将CAM转换为numpy数组
    attention_weights = attention_weights.squeeze().detach().numpy()
    # 归一化CAM以便可视化
    attention_weights = (attention_weights - np.min(attention_weights)) / (np.max(attention_weights) - np.min(attention_weights))
    return attention_weights


def show_image(image):
    image = image.detach()
    image = np.squeeze(image)  # 去除多余的维度，变为 (28, 28)
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # 关闭坐标轴显示
    plt.show()




def calculate_CAM(sample_image,attention_weights):
    # 获取目标层的特征图
    activations = model.conv1(sample_image)
    target_activations = activations[0, :, :, :]
    grad_cam = attention_weights * target_activations.detach().numpy()
    grad_cam = np.maximum(grad_cam, 0)  # ReLU操作

    # 将Grad-CAM映射到原始图像大小
    grad_cam = cv2.resize(grad_cam.sum(axis=0), (28, 28))
    grad_cam = (grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam))
    grad_cam = np.squeeze(grad_cam)
    return grad_cam


def get_img(target_class):
    index_of_2 = next(i for i, (img, label) in enumerate(mnist_test) if label == target_class)
    # 读取图像并进行预处理
    img, _ = mnist_test[index_of_2]
    sample_image = Variable(img.unsqueeze(0))
    return sample_image

def test_fgsm_t(model, test_loader,target_class, epsilon):
    model.eval()
    correct = 0
    correct2 = 0
    total = 0
    running_loss=0.0
    for images, labels in test_loader:
        sample_image=get_img(target_class)

        #计算梯度权重
        gradients=GET_Grad(model, sample_image,target_class)
        #获得权重矩阵
        attention_weights=calculate_weights(gradients)
        # 计算Grad-CAM
        grad_cam=calculate_CAM(sample_image,attention_weights)
        
        #将Grad-CAM转换为RGB格式的热力图heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
        heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

        #将原图sample_image转换为RGB格式
        img_array = np.array(sample_image)
        img_array = np.squeeze(img_array)  # 去除多余的维度，变为 (28, 28)
        # 将灰度图扩展为具有三个相同通道的彩色图像
        img_rgb = np.expand_dims(img_array, axis=-1)  # 扩展维度，变为 (28, 28, 1)
        img_rgb = np.repeat(img_rgb, 3, axis=-1)  # 重复三次通道，变为 (28, 28, 3)
        #叠加
        result = heatmap * 0.003 + img_rgb * 0.5
        #注意力图可视化
        #plt_show(sample_image,heatmap,result)

        #干扰训练
        target = torch.full_like(labels, fill_value=target_class)
        images = images.requires_grad_()
        output = model(images)
        _, predicted1 = torch.max(output.data, 1)
        loss = nn.CrossEntropyLoss()(output, target)
        model.zero_grad()
        perturbed_images=images
        for i in range(0,1):
            output = model(perturbed_images)

            loss = nn.CrossEntropyLoss()(output, labels)
            loss.backward()  # 计算梯度
            running_loss=loss.item()
            # 获取输入图像的梯度
            data_grad = images.grad.data
            attention_weights = calculate_weights(data_grad)
            attention_weights_tensor = torch.from_numpy(attention_weights)
            # 仅对卷积层权值矩阵中前25%权重元素大小的像素点进行扰
            _, indices = torch.topk(attention_weights_tensor.view(-1), int(0.35 * attention_weights_tensor.numel()),largest=True)
            mask = torch.zeros_like(attention_weights_tensor.view(-1))
            mask[indices] = 2
            mask = mask.view_as(attention_weights_tensor)
            data_grad *= mask

            #show_image(images[1])
            # FGSM攻击
            perturbed_images = fgsm_t_attack(model, images, data_grad, epsilon)
            #show_image(perturbed_images[1])
        # 在对抗样本上进行推理
        output = model(perturbed_images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        correct2 += (predicted == target_class).sum().item()

        #print(f"真实{labels}\n原预测{predicted1}\n 干扰预测{predicted} \n")

    accuracy = correct / total
    accuracy2 = correct2 / total
    print(f"epsilon={epsilon}: {accuracy * 100:.2f}%")
    #print(f"target={target_class} Attack-Accuracy: {accuracy2 * 100:.2f}%")


# 测试模型在原始测试集上的性能
print("Accuracy:")
test_fgsm_t(model, test_loader,8, epsilon=0.4)

# 测试模型在对抗样本上的性能
print("\nAccuracy(Adversarial):")
test_fgsm_t(model, test_loader,2, epsilon=0.5)

# 测试模型在对抗样本上的性能
print("\nAccuracy(Adversarial):")
test_fgsm_t(model, test_loader,1, epsilon=0.1)

# 测试模型在对抗样本上的性能
print("\nAccuracy(Adversarial):")
test_fgsm_t(model, test_loader,2, epsilon=0.12)

# 测试模型在对抗样本上的性能
print("\nAccuracy(Adversarial):")
test_fgsm_t(model, test_loader,2, epsilon=0.17)

# 测试模型在对抗样本上的性能
print("\nAccuracy(Adversarial):")
test_fgsm_t(model, test_loader,2, epsilon=0.2)

# 测试模型在对抗样本上的性能
print("\nAccuracy(Adversarial):")
test_fgsm_t(model, test_loader, 2,epsilon=0.25)

# 测试模型在对抗样本上的性能
print("\nAccuracy(Adversarial):")
test_fgsm_t(model, test_loader,2, epsilon=0.3)

# 测试模型在对抗样本上的性能
print("\nAccuracy(Adversarial):")
test_fgsm_t(model, test_loader,2, epsilon=2)



