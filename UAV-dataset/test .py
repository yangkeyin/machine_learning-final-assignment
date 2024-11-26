import os
import torch
import matplotlib.pyplot as plt
from baseline import UAVDetectionModel, ImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime 
import pandas as pd

test_dir = './dataset1/test'


def save_results(predictions, labels, save_dir='./results'):
    # 创建保存结果的目录
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存混淆矩阵
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    # 保存混淆矩阵图像
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{timestamp}.png'))
    plt.close()
    
    # 保存混淆矩阵数据
    cm_df = pd.DataFrame(
        cm, 
        index=['背景', '无人机'], 
        columns=['预测_背景', '预测_无人机']
    )
    cm_df.to_csv(os.path.join(save_dir, f'confusion_matrix_{timestamp}.csv'))
    
    # 生成分类报告
    report = classification_report(
        labels, 
        predictions,
        target_names=['背景', '无人机'],
        digits=4,
        output_dict=True
    )
    
    # 将分类报告转换为DataFrame并保存
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(save_dir, f'classification_report_{timestamp}.csv'))
    
    # 保存原始预测结果
    results_df = pd.DataFrame({
        'True_Label': labels,
        'Predicted_Label': predictions
    })
    results_df.to_csv(os.path.join(save_dir, f'predictions_{timestamp}.csv'), index=False)
    
    return cm_df, report_df


def evaluate_model(model, test_loader, device):

    model.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    wrong_predictions = []
    
    with torch.no_grad():
        for images, labels, img_paths in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 记录错误预测的图片路径
            mask = (predicted != labels)
            if mask.any():
                wrong_paths = [img_paths[i] for i in range(len(mask)) if mask[i]]
                wrong_pred = [predicted[i].item() for i in range(len(mask)) if mask[i]]
                wrong_true = [labels[i].item() for i in range(len(mask)) if mask[i]]
                wrong_predictions.extend(list(zip(wrong_paths, wrong_pred, wrong_true)))
            
            # 保存预测结果和真实标签
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_predictions, all_labels

def main():
    # 设置设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"using device:{device} ")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    test_dataset = ImageDataset(os.path.join(test_dir, 'UAV'), os.path.join(test_dir, 'background'), transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)



    model = UAVDetectionModel()
    # 加载最佳模型
    model = UAVDetectionModel().to(device)
    checkpoint = torch.load('best_model_efficientnet.pth', map_location=device)
    if 'state_dict' in checkpoint:
        # 如果模型状态保存在 'state_dict' 键中
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # 直接加载状态字典
        model.load_state_dict(checkpoint)


    # 评估模型
    predictions, labels = evaluate_model(model, test_loader, device)
    
    cm_df, report_df = save_results(predictions, labels)
    
    # 显示结果
    print("\n混淆矩阵:")
    print(cm_df)
    print("\n分类报告:")
    print(report_df)
    
    # 显示保存位置
    print("\n结果已保存在 ./results 目录下")
    
if __name__ == '__main__':
    main()