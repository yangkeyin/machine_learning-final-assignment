# ... 前面的代码保持不变 ...

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
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
    
    accuracy = 100 * correct / total
    
    print(f'\n测试集评估结果:')
    print(f'总样本数: {total}')
    print(f'正确预测数: {correct}')
    print(f'测试集准确率: {accuracy:.2f}%')
    
    # 打印错误预测的详情
    if wrong_predictions:
        print("\n错误预测的样本:")
        for path, pred, true in wrong_predictions[:10]:  # 只显示前10个错误预测
            print(f"图片: {path}")
            print(f"预测标签: {'无人机' if pred == 1 else '背景'}")
            print(f"真实标签: {'无人机' if true == 1 else '背景'}")
            print("---")
    
    return accuracy, wrong_predictions

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 在测试集上评估模型
test_accuracy, wrong_predictions = evaluate_model(model, test_loader)

# 计算并显示混淆矩阵
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def plot_confusion_matrix(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['背景', '无人机'],
                              digits=4))

# 绘制混淆矩阵和分类报告
plot_confusion_matrix(model, test_loader)