"""
评估模块
实现模型评估和验证功能
"""

import numpy as np
from loss_function import get_loss_function


def accuracy(predictions, targets):
    """
    计算准确率
    
    Args:
        predictions: 预测结果 (batch_size, num_classes) 或 (batch_size,)
        targets: 真实标签 (batch_size, num_classes) 或 (batch_size,)
    
    Returns:
        accuracy: 准确率
    """
    if predictions.ndim > 1:
        # 如果是概率分布，取最大值的索引
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions
    
    if targets.ndim > 1:
        # 如果是one-hot编码，取最大值的索引
        true_classes = np.argmax(targets, axis=1)
    else:
        true_classes = targets
    
    correct = np.sum(pred_classes == true_classes)
    total = len(pred_classes)
    
    return correct / total


def precision_recall_f1(predictions, targets, average='macro'):
    """
    计算精确率、召回率和F1分数
    
    Args:
        predictions: 预测结果 (batch_size, num_classes) 或 (batch_size,)
        targets: 真实标签 (batch_size, num_classes) 或 (batch_size,)
        average: 平均方式 ('macro', 'micro', 'weighted')
    
    Returns:
        precision: 精确率
        recall: 召回率
        f1: F1分数
    """
    if predictions.ndim > 1:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions
    
    if targets.ndim > 1:
        true_classes = np.argmax(targets, axis=1)
    else:
        true_classes = targets
    
    num_classes = len(np.unique(np.concatenate([pred_classes, true_classes])))
    
    # 计算混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(pred_classes)):
        confusion_matrix[true_classes[i], pred_classes[i]] += 1
    
    if average == 'micro':
        # Micro平均：所有类别的TP、FP、FN的总和
        tp = np.sum(np.diag(confusion_matrix))
        fp = np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
        fn = np.sum(confusion_matrix, axis=1) - np.diag(confusion_matrix)
        
        precision = tp / (tp + np.sum(fp))
        recall = tp / (tp + np.sum(fn))
        f1 = 2 * precision * recall / (precision + recall)
    
    elif average == 'macro':
        # Macro平均：每个类别指标的算术平均
        precisions = []
        recalls = []
        f1s = []
        
        for i in range(num_classes):
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)
    
    elif average == 'weighted':
        # Weighted平均：按样本数量加权
        precisions = []
        recalls = []
        f1s = []
        weights = []
        
        for i in range(num_classes):
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            
            weight = np.sum(confusion_matrix[i, :])
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        precision = np.sum(np.array(precisions) * weights)
        recall = np.sum(np.array(recalls) * weights)
        f1 = np.sum(np.array(f1s) * weights)
    
    else:
        raise ValueError(f"不支持的average参数: {average}")
    
    return precision, recall, f1


def confusion_matrix(predictions, targets, class_names=None):
    """
    计算混淆矩阵
    
    Args:
        predictions: 预测结果 (batch_size, num_classes) 或 (batch_size,)
        targets: 真实标签 (batch_size, num_classes) 或 (batch_size,)
        class_names: 类别名称列表
    
    Returns:
        confusion_matrix: 混淆矩阵
        class_names: 类别名称
    """
    if predictions.ndim > 1:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions
    
    if targets.ndim > 1:
        true_classes = np.argmax(targets, axis=1)
    else:
        true_classes = targets
    
    # 获取所有类别
    all_classes = np.unique(np.concatenate([pred_classes, true_classes]))
    num_classes = len(all_classes)
    
    # 创建混淆矩阵
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(pred_classes)):
        true_idx = np.where(all_classes == true_classes[i])[0][0]
        pred_idx = np.where(all_classes == pred_classes[i])[0][0]
        cm[true_idx, pred_idx] += 1
    
    # 类别名称
    if class_names is None:
        class_names = [str(i) for i in all_classes]
    else:
        class_names = [class_names[i] for i in all_classes]
    
    return cm, class_names


def evaluate_model(network, data, labels, loss_function_name='cross_entropy', batch_size=64):
    """
    评估模型性能
    
    Args:
        network: 神经网络模型
        data: 输入数据 (num_samples, input_size)
        labels: 标签 (num_samples, num_classes) 或 (num_samples,)
        loss_function_name: 损失函数名称
        batch_size: 批次大小
    
    Returns:
        results: 评估结果字典
    """
    # 获取损失函数
    loss_function = get_loss_function(loss_function_name)
    
    # 分批处理数据
    num_samples = len(data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    total_loss = 0
    total_accuracy = 0
    all_predictions = []
    all_labels = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        batch_data = data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # 前向传播
        predictions = network.predict(batch_data)
        
        # 计算损失
        loss = loss_function.forward(predictions, batch_labels)
        total_loss += loss
        
        # 计算准确率
        acc = accuracy(predictions, batch_labels)
        total_accuracy += acc
        
        # 保存预测结果
        all_predictions.append(predictions)
        all_labels.append(batch_labels)
    
    # 合并所有预测结果
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels) if all_labels[0].ndim > 1 else np.concatenate(all_labels)
    
    # 计算平均指标
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    # 计算精确率、召回率、F1分数
    precision, recall, f1 = precision_recall_f1(all_predictions, all_labels)
    
    # 计算混淆矩阵
    cm, class_names = confusion_matrix(all_predictions, all_labels)
    
    results = {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'class_names': class_names,
        'predictions': all_predictions,
        'true_labels': all_labels
    }
    
    return results


def print_evaluation_results(results, dataset_name='Dataset'):
    """
    打印评估结果
    
    Args:
        results: 评估结果字典
        dataset_name: 数据集名称
    """
    print(f"\n{dataset_name} 评估结果:")
    print("=" * 50)
    print(f"损失值: {results['loss']:.4f}")
    print(f"准确率: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    # 只有当precision与accuracy差异显著时才显示precision
    accuracy_val = results['accuracy']
    precision_val = results['precision']
    
    # 如果差异大于0.01才显示precision
    if abs(precision_val - accuracy_val) > 0.01:
        print(f"精确率: {results['precision']:.4f}")
    
    # 不使用召回率和F1-score，因为对于手写数字识别任务不是必需的
    print("=" * 50)


def find_misclassified_samples(results, num_samples=10):
    """
    找出错误分类的样本
    
    Args:
        results: 评估结果字典
        num_samples: 返回的样本数量
    
    Returns:
        misclassified_indices: 错误分类样本的索引
        misclassified_predictions: 错误分类的预测结果
        misclassified_labels: 错误分类的真实标签
    """
    predictions = results['predictions']
    true_labels = results['true_labels']
    
    if predictions.ndim > 1:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions
    
    if true_labels.ndim > 1:
        true_classes = np.argmax(true_labels, axis=1)
    else:
        true_classes = true_labels
    
    # 找出错误分类的样本
    misclassified_mask = pred_classes != true_classes
    misclassified_indices = np.where(misclassified_mask)[0]
    
    # 随机选择一些错误分类的样本
    if len(misclassified_indices) > num_samples:
        selected_indices = np.random.choice(
            misclassified_indices, 
            size=num_samples, 
            replace=False
        )
    else:
        selected_indices = misclassified_indices
    
    return selected_indices, pred_classes[selected_indices], true_classes[selected_indices]


def class_wise_accuracy(results):
    """
    计算每个类别的准确率
    
    Args:
        results: 评估结果字典
    
    Returns:
        class_accuracies: 每个类别的准确率
    """
    cm = results['confusion_matrix']
    class_names = results['class_names']
    
    class_accuracies = {}
    for i in range(len(class_names)):
        correct = cm[i, i]
        total = np.sum(cm[i, :])
        accuracy = correct / total if total > 0 else 0
        class_accuracies[class_names[i]] = accuracy
    
    return class_accuracies