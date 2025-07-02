import torchvision.models as models
from torchinfo import summary
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 配置GPU，这里有两种方式
## 使用os.environ
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

## 配置其他超参数，如batch_size, num_workers, learning rate, 以及总的epochs
batch_size =512 #512
num_workers = 0   # 对于Windows用户，这里应设置为0，否则会出现多线程错误
lr = 1e-3
epochs = 50

# 首先设置数据变换
from torchvision import transforms

image_size = 28
# 为训练集定义一个带数据增强的 transform
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(32),

    # --- 数据增强 ---
    transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
    transforms.RandomRotation(10),       # 随机旋转 (-10, 10度)
    # ----------------

    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # 归一化，对于单通道灰度图转三通道，用(0.5,)即可
])

# 为测试集定义一个没有数据增强的 transform
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class FMDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.images = df.iloc[:, 1:].values.astype(np.uint8)
        self.labels = df.iloc[:, 0].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28, 1)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image / 255., dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

train_df = pd.read_csv("FashionMNIST/fashion-mnist_train.csv")
test_df = pd.read_csv("FashionMNIST/fashion-mnist_test.csv")
train_data = FMDataset(train_df, train_transform)
test_data = FMDataset(test_df, test_transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 记录损失值
train_losses = []
val_losses = []
val_accs = []
resnet = models.resnet18() # 实例化模型

resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
resnet.maxpool = nn.Identity() # 相当于去掉了最大池化层
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)
resnet = resnet.cuda()
summary(resnet, (batch_size, 3, 28, 28)) # 1：batch_size 3:图片的通道数 224: 图片的高宽

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.AdamW(resnet.parameters(), lr = lr, weight_decay=1e-4)

def train(epoch):
    resnet.train()
    train_loss = 0
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = resnet(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
    train_losses.append(train_loss)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
def val(epoch):
    resnet.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            output = resnet(data)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
    val_loss = val_loss/len(test_loader.dataset)
    val_losses.append(val_loss)  # 添加验证损失到列表
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
    val_accs.append(acc)  # 添加验证准确率到列表
    print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))
for epoch in range(1, epochs+1):
    train(epoch)
    val(epoch)

torch.save(resnet, "FashionModelResnet1850.pth")
# model = torch.load("FashionModel.pth")

# 绘制损失曲线图
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_accs, label='Validation Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()