import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt

# 定义数据集路径
train_dir = './dataset1/train'
val_dir = './dataset1/val'
test_dir = './dataset1/test'


# 自定义数据集
class ImageDataset(Dataset):
    def __init__(self, uav_dir, background_dir, transform=None):
        self.uav_images = [os.path.join(uav_dir, img) for img in os.listdir(uav_dir)]
        self.background_images = [os.path.join(background_dir, img) for img in os.listdir(background_dir)]
        self.images = self.uav_images + self.background_images
        self.labels = [1] * len(self.uav_images) + [0] * len(self.background_images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path  


# ... 前面的代码保持不变 ...

# 定义CNN模型
class UAVDetectionModel(nn.Module):
    def __init__(self):
        super(UAVDetectionModel, self).__init__()
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 二分类：无人机/背景
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # 验证
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        train_losses.append(epoch_loss)
        val_accuracies.append(epoch_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Accuracy: {epoch_acc:.2f}%')
        
        # 保存最佳模型
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return train_losses, val_accuracies


# 数据增强和转换
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 创建数据集和数据加载器
train_dataset = ImageDataset(os.path.join(train_dir, 'UAV'), os.path.join(train_dir, 'background'), transform=transform)
val_dataset = ImageDataset(os.path.join(val_dir, 'UAV'), os.path.join(val_dir, 'background'), transform=transform)
test_dataset = ImageDataset(os.path.join(test_dir, 'UAV'), os.path.join(test_dir, 'background'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

# 创建模型实例
model = UAVDetectionModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer)

# 绘制训练过程
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('训练损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies)
plt.title('验证准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.show()
