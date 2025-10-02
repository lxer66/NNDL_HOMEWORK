"""
可视化模块
实现训练过程和测试结果的可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, 
                         final_results=None, save_path=None):
    """
    绘制训练历史曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accuracies: 训练准确率列表
        val_accuracies: 验证准确率列表
        final_results: 最终评估结果字典，包含loss, accuracy, precision, recall, f1_score等指标
        save_path: 保存路径
    """
    # 创建一个更大的图形以容纳指标文本
    fig = plt.figure(figsize=(18, 5))
    
    # 使用GridSpec创建不等大的子图
    gs = plt.GridSpec(1, 3, width_ratios=[2, 2, 1], figure=fig)
    
    # 损失曲线子图
    ax1 = fig.add_subplot(gs[0])
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线子图
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 指标文本子图
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')  # 隐藏坐标轴
    
    if final_results is not None:
        # 准备文本内容，区分训练和验证集指标
        metrics_text = "Final Results:\n\n"
        
        # 损失
        if 'train_loss' in final_results and 'val_loss' in final_results:
            metrics_text += f"Training Loss:   {final_results['train_loss']:.4f}\n"
            metrics_text += f"Validation Loss: {final_results['val_loss']:.4f}\n\n"
        else:
            metrics_text += f"Loss:            {final_results['loss']:.4f}\n\n"
        
        # 准确率
        if 'train_accuracy' in final_results and 'val_accuracy' in final_results:
            metrics_text += f"Training Accuracy:   {final_results['train_accuracy']:.4f}\n"
            metrics_text += f"Validation Accuracy: {final_results['val_accuracy']:.4f}\n\n"
        else:
            metrics_text += f"Accuracy:            {final_results['accuracy']:.4f}\n\n"
        
        # 精度(Precision) - 根据要求，如果与准确率差异不大则移除
        # 只有当precision与accuracy差异显著时才显示
        if 'precision' in final_results:
            # 获取准确率用于比较
            accuracy_val = final_results.get('accuracy', 0)
            if 'val_accuracy' in final_results:
                accuracy_val = final_results['val_accuracy']
            
            precision_val = final_results['precision']
            
            # 如果差异大于0.01才显示precision
            if abs(precision_val - accuracy_val) > 0.01:
                metrics_text += f"Precision:    {precision_val:.4f}\n\n"
        
        # 在文本子图中添加文本
        ax3.text(0.1, 0.5, metrics_text, transform=ax3.transAxes, fontsize=12,
                verticalalignment='center', fontfamily='monospace')
    else:
        ax3.text(0.5, 0.5, "No final results available", transform=ax3.transAxes, 
                fontsize=12, horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.close(fig)  # 关闭图形而不是显示它
    # plt.show()


def plot_confusion_matrix(confusion_matrix, class_names, title='Confusion Matrix', save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        confusion_matrix: 混淆矩阵
        class_names: 类别名称
        title: 图表标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 计算百分比
    cm_percent = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100
    
    # 绘制热力图
    im = ax.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 设置坐标轴
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # 旋转标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在每个格子中添加数值
    fmt = '.1f'
    thresh = cm_percent.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, f'{confusion_matrix[i, j]}\n({cm_percent[i, j]:.1f}%)',
                   ha="center", va="center",
                   color="white" if cm_percent[i, j] > thresh else "black",
                   fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close(fig)  # 关闭图形而不是显示它
    # plt.show()


def plot_class_accuracy(class_accuracies, title='Class-wise Accuracy', save_path=None):
    """
    绘制各类别准确率
    
    Args:
        class_accuracies: 各类别准确率字典
        title: 图表标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    # 创建柱状图
    bars = ax.bar(classes, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # 在每个柱子上添加数值
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 旋转x轴标签
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class accuracy plot saved to: {save_path}")
    
    plt.close(fig)  # 关闭图形而不是显示它
    # plt.show()


def plot_misclassified_samples(images, predictions, true_labels, class_names, 
                              num_samples=25, save_path=None):
    """
    绘制错误分类样本
    
    Args:
        images: 图像数据 (num_samples, 784)
        predictions: 预测结果
        true_labels: 真实标签
        class_names: 类别名称
        num_samples: 显示的样本数量
        save_path: 保存路径
    """
    # 重塑图像为28x28
    if images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)
    
    # 获取预测类别
    if predictions.ndim > 1:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions
    
    if true_labels.ndim > 1:
        true_classes = np.argmax(true_labels, axis=1)
    else:
        true_classes = true_labels
    
    # 计算网格大小
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        ax = axes[i]
        
        # 显示图像
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'True: {class_names[true_classes[i]]}, '
                    f'Pred: {class_names[pred_classes[i]]}', 
                    fontsize=10, color='red')
        ax.axis('off')
    
    # 隐藏多余的子图
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Misclassified Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Misclassified samples chart saved to: {save_path}")
    
    plt.close(fig)  # 关闭图形而不是显示它
    # plt.show()


def plot_weight_distribution(network, save_path=None):
    """
    绘制权重分布
    
    Args:
        network: 神经网络模型
        save_path: 保存路径
    """
    weights, biases = network.get_weights()
    
    fig, axes = plt.subplots(2, len(weights), figsize=(15, 8))
    if len(weights) == 1:
        axes = axes.reshape(2, 1)
    
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        # 权重分布
        ax_weight = axes[0, i]
        ax_weight.hist(weight.flatten(), bins=50, alpha=0.7, color='blue')
        ax_weight.set_title(f'Layer {i+1} Weight Distribution')
        ax_weight.set_xlabel('Weight Value')
        ax_weight.set_ylabel('Frequency')
        ax_weight.grid(True, alpha=0.3)
        
        # 偏置分布
        ax_bias = axes[1, i]
        ax_bias.hist(bias.flatten(), bins=30, alpha=0.7, color='red')
        ax_bias.set_title(f'Layer {i+1} Bias Distribution')
        ax_bias.set_xlabel('Bias Value')
        ax_bias.set_ylabel('Frequency')
        ax_bias.grid(True, alpha=0.3)
    
    plt.suptitle('Network Weight and Bias Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Weight distribution plot saved to: {save_path}")
    
    plt.close(fig)  # 关闭图形而不是显示它
    # plt.show()


def generate_text_report(train_results, test_results, config, save_dir):
    """
    生成文本报告
    
    Args:
        train_results: 训练结果
        test_results: 测试结果
        config: 配置
        save_dir: 保存目录
    """
    report_path = os.path.join(save_dir, 'report.txt')
    
    with open(report_path, 'w') as f:
        f.write("Neural Network Training Report\n")
        f.write("=" * 50 + "\n\n")
        
        # 配置信息
        f.write("Configuration:\n")
        f.write(f"  Network structure: {config.layer_sizes}\n")
        f.write(f"  Activation function: {config.activation}\n")
        f.write(f"  Loss function: {config.loss_function}\n")
        f.write(f"  Optimizer: {config.optimizer}\n")
        f.write(f"  Regularization: {config.regularization}\n")
        f.write(f"  Epochs: {config.epochs}\n")
        f.write(f"  Batch size: {config.batch_size}\n")
        f.write(f"  Learning rate: {config.learning_rate}\n\n")
        
        # 训练结果
        f.write("Training Results:\n")
        f.write(f"  Final training loss: {train_results['train_losses'][-1]:.6f}\n")
        f.write(f"  Final validation loss: {train_results['val_losses'][-1]:.6f}\n")
        f.write(f"  Final training accuracy: {train_results['train_accuracies'][-1]:.6f}\n")
        f.write(f"  Final validation accuracy: {train_results['val_accuracies'][-1]:.6f}\n\n")
        
        # 测试结果
        f.write("Test Results:\n")
        f.write(f"  Loss: {test_results['loss']:.6f}\n")
        f.write(f"  Accuracy: {test_results['accuracy']:.6f}\n")
        
        # 只有当precision与accuracy差异显著时才记录precision
        accuracy_val = test_results['accuracy']
        precision_val = test_results['precision']
        
        # 如果差异大于0.01才记录precision
        if abs(precision_val - accuracy_val) > 0.01:
            f.write(f"  Precision: {test_results['precision']:.6f}\n")
    
    print(f"Text report saved to: {report_path}")


def visualize_all(train_results, test_results, config, save_dir='pictures'):
    """
    可视化所有结果
    
    Args:
        train_results: 训练结果
        test_results: 测试结果
        config: 配置
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 训练历史
    plot_training_history(
        train_results['train_losses'],
        train_results['val_losses'],
        train_results['train_accuracies'],
        train_results['val_accuracies'],
        os.path.join(save_dir, 'training_history.png')
    )
    
    # 2. 测试集混淆矩阵
    plot_confusion_matrix(
        test_results['confusion_matrix'],
        test_results['class_names'],
        'Test Set Confusion Matrix',
        os.path.join(save_dir, 'test_confusion_matrix.png')
    )
    
    # 3. 类别准确率
    from evaluate import class_wise_accuracy
    class_acc = class_wise_accuracy(test_results)
    plot_class_accuracy(
        class_acc,
        'Test Set Class Accuracies',
        os.path.join(save_dir, 'test_class_accuracy.png')
    )
    
    # 4. 错误分类样本
    from evaluate import find_misclassified_samples
    misclassified_indices, pred_classes, true_classes = find_misclassified_samples(
        test_results, num_samples=16
    )
    
    if len(misclassified_indices) > 0:
        # 这需要原始图像数据，在实际使用中需要从测试数据中获取
        print("Visualizing misclassified samples requires original image data")
    
    # 5. 权重分布
    plot_weight_distribution(
        train_results['network'],
        os.path.join(save_dir, 'weight_distribution.png')
    )
    
    # 6. 生成文本报告
    generate_text_report(train_results, test_results, config, save_dir)