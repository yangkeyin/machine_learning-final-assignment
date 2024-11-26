import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models

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
    def __init__(self, num_classes=2):
        super(UAVDetectionModel, self).__init__()
        # Use EfficientNet-B0 instead
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Freeze only the first few layers
        ct = 0
        for child in self.model.features.children():
            ct += 1
            if ct < 3:  # Only freeze first 2 blocks
                for param in child.parameters():
                    param.requires_grad = False
                    
        # Modify classifier
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
        model = model.to(device)
    else:
        device = torch.device("cpu")
        print("MPS device not found, using CPU")
        model = model.to(device)
    
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    

    best_val_acc = 0.0
    patience = 2
    train_losses = []
    val_accuracies = []
    # 添加 L2 正则化参数
    l2_lambda = 0.01
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            # 计算主损失
            main_loss = criterion(outputs, labels)
            
            # 添加 L2 正则化
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss = main_loss + l2_lambda * l2_reg
            loss.backward()
              # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # 更新学习率
        scheduler.step()
         # 早停检查
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            best_epoch = epoch
            torch.save({
                'model_type': 'efficientnet',  # 或 'resnet'
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc,
            }, 'best_model_efficientnet.pth')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping triggered. Best epoch was {best_epoch+1}')
                break

    
    return train_losses, val_accuracies

if __name__ == '__main__':
    # use resnet standard pretrained
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建数据集和数据加载器
    train_dataset = ImageDataset(os.path.join(train_dir, 'UAV'), os.path.join(train_dir, 'background'), transform=transform)
    val_dataset = ImageDataset(os.path.join(val_dir, 'UAV'), os.path.join(val_dir, 'background'), transform=transform)
    #test_dataset = ImageDataset(os.path.join(test_dir, 'UAV'), os.path.join(test_dir, 'background'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    # 创建模型实例
    model = UAVDetectionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,  # 降低初始学习率
        weight_decay=0.01,  # 增加权重衰减
        betas=(0.9, 0.999)
    )


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


