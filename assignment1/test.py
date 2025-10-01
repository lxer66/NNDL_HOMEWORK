"""
测试模块
独立的模型测试和评估模块
"""

import os
import argparse
import numpy as np
from network import NeuralNetwork
from loss_function import get_loss_function
from preprocess import preprocess_mnist
from evaluate import evaluate_model, print_evaluation_results, find_misclassified_samples, class_wise_accuracy
from visualize import plot_confusion_matrix, plot_class_accuracy, plot_misclassified_samples, plot_weight_distribution, plot_training_history


def parse_test_args():
    """解析测试命令行参数"""
    parser = argparse.ArgumentParser(description='MNIST神经网络测试')
    
    # 必需参数
    parser.add_argument('--model_path', type=str, default='models/sgd_cross_entropy_l2_final.npy',
                       help='模型权重文件路径')
    
    # 可选参数
    parser.add_argument('--loss_function', type=str, default='cross_entropy',
                       choices=['cross_entropy', 'mse', 'l1_loss'],
                       help='损失函数类型')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='测试批次大小')
    parser.add_argument('--visualize', action='store_true',
                       help='是否生成可视化结果')
    parser.add_argument('--save_dir', type=str, default='test_results',
                       help='结果保存目录')
    parser.add_argument('--num_misclassified', type=int, default=16,
                       help='显示的错误分类样本数量')
    parser.add_argument('--layer_sizes', nargs='+', type=int, default=[784, 216, 128, 10],
                       help='网络层结构（需要与训练时一致）')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'tanh', 'sigmoid'],
                       help='激活函数类型')
    
    return parser.parse_args()


class ModelTester:
    """模型测试器"""
    
    def __init__(self, args):
        """
        Args:
            args: 命令行参数
        """
        self.args = args
        self.network = None
        self.test_data = None
        self.test_labels = None
        
        # 创建保存目录
        os.makedirs(args.save_dir, exist_ok=True)
    
    def load_model(self):
        """加载模型"""
        print(f"加载模型: {self.args.model_path}")
        
        if not os.path.exists(self.args.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.args.model_path}")
        
        # 创建网络结构
        self.network = NeuralNetwork(
            layer_sizes=self.args.layer_sizes,
            activation=self.args.activation,
            weight_init='he'  # 这个参数在测试时不重要
        )
        
        # 加载权重
        self.network.load_weights(self.args.model_path)
        
        print(f"模型加载成功！")
        print(f"网络结构: {self.args.layer_sizes}")
        print(f"参数数量: {self.network.get_parameter_count():,}")
    
    def prepare_data(self):
        """准备测试数据"""
        print("准备测试数据...")
        
        # 预处理数据
        data = preprocess_mnist(
            train_ratio=0.8,
            val_ratio=0.2,
            normalize=True,
            one_hot=True
        )
        
        self.test_data = data['test_images']
        self.test_labels = data['test_labels']
        
        print(f"测试集大小: {len(self.test_data)}")
        print(f"数据形状: {self.test_data.shape}")
        print(f"标签形状: {self.test_labels.shape}")
    
    def test_model(self):
        """测试模型"""
        print("开始测试模型...")
        
        # 评估模型
        results = evaluate_model(
            self.network, 
            self.test_data, 
            self.test_labels,
            self.args.loss_function,
            self.args.batch_size
        )
        
        # 打印结果
        print_evaluation_results(results, "测试集")
        
        return results
    
    def analyze_results(self, results):
        """分析测试结果"""
        print("\n详细分析:")
        print("=" * 50)
        
        # 类别准确率分析
        class_acc = class_wise_accuracy(results)
        print("各类别准确率:")
        for class_name, acc in class_acc.items():
            print(f"  数字 {class_name}: {acc:.4f} ({acc*100:.2f}%)")
        
        # 找出最差和最好的类别
        best_class = max(class_acc, key=class_acc.get)
        worst_class = min(class_acc, key=class_acc.get)
        print(f"\n最佳分类: 数字 {best_class} ({class_acc[best_class]*100:.2f}%)")
        print(f"最差分类: 数字 {worst_class} ({class_acc[worst_class]*100:.2f}%)")
        
        # 错误分类样本分析
        misclassified_indices, pred_classes, true_classes = find_misclassified_samples(
            results, num_samples=self.args.num_misclassified
        )
        
        print(f"\n错误分类样本分析:")
        print(f"  总错误样本数: {len(misclassified_indices)}")
        print(f"  错误率: {len(misclassified_indices) / len(results['true_labels']) * 100:.2f}%")
        
        if len(misclassified_indices) > 0:
            print(f"  前{min(5, len(misclassified_indices))}个错误分类:")
            for i in range(min(5, len(misclassified_indices))):
                print(f"    样本 {misclassified_indices[i]}: 真实={true_classes[i]}, 预测={pred_classes[i]}")
    
    def generate_visualizations(self, results):
        """生成可视化结果"""
        # 默认生成可视化结果，除非明确禁用
        # 注意：这里我们移除了对args.visualize的检查，使可视化成为默认行为
        
        print("生成可视化结果...")
        
        # 从模型路径提取命名前缀
        model_filename = os.path.basename(self.args.model_path)
        if '_final.npy' in model_filename:
            name_prefix = model_filename.replace('_final.npy', '')
        elif '.npy' in model_filename:
            name_prefix = model_filename.replace('.npy', '')
        else:
            name_prefix = model_filename
            
        # 创建test_results目录及子目录
        output_dir = os.path.join('test_results', name_prefix)
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 绘制混淆矩阵
        cm = results['confusion_matrix']
        class_names = [str(i) for i in range(10)]  # MNIST有10个数字类别
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_path=cm_path)
        
        # 2. 绘制各类别准确率
        class_acc = class_wise_accuracy(results)
        acc_path = os.path.join(output_dir, "class_accuracy.png")
        plot_class_accuracy(class_acc, title='Class-wise Accuracy', save_path=acc_path)
        
        # 3. 显示部分错误分类样本
        if 'predictions' in results and 'true_labels' in results:
            # 获取错误分类样本
            misclassified_indices, pred_classes, true_classes = find_misclassified_samples(
                results, num_samples=self.args.num_misclassified
            )
            
            if len(misclassified_indices) > 0:
                # 获取前几个错误分类样本
                num_samples = min(self.args.num_misclassified, len(misclassified_indices))
                misclassified_images = self.test_data[misclassified_indices[:num_samples]]
                misclassified_preds = results['predictions'][misclassified_indices[:num_samples]]
                misclassified_labels = results['true_labels'][misclassified_indices[:num_samples]]
                
                # 绘制错误分类样本
                misclassified_path = os.path.join(output_dir, "misclassified_samples.png")
                plot_misclassified_samples(
                    misclassified_images, 
                    misclassified_preds, 
                    misclassified_labels, 
                    class_names,
                    num_samples=num_samples,
                    save_path=misclassified_path
                )
        
        # 4. 绘制网络权重分布
        weight_dist_path = os.path.join(output_dir, "weight_distribution.png")
        plot_weight_distribution(self.network, save_path=weight_dist_path)
        
        # 5. 保存测试结果到文本文件
        self.save_test_results_to_txt(results, name_prefix)
        
        print(f"Visualization results saved to: {output_dir}")
    
    def save_test_results_to_txt(self, results, name_prefix):
        """将测试结果保存到文本文件"""
        # 创建test_results目录及子目录
        output_dir = os.path.join('test_results', name_prefix)
        os.makedirs(output_dir, exist_ok=True)
        
        # 构造文件路径
        txt_path = os.path.join(output_dir, "test_results.txt")
        
        # 打开文件并写入结果
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("MNIST神经网络测试结果\n")
            f.write("=" * 50 + "\n\n")
            
            # 写入基本评估指标
            f.write("基本评估指标:\n")
            f.write(f"  测试损失 (Loss):     {results['loss']:.6f}\n")
            f.write(f"  测试准确率 (Accuracy): {results['accuracy']:.6f}\n")
            
            # 只有当precision与accuracy差异显著时才显示precision
            accuracy_val = results['accuracy']
            precision_val = results['precision']
            
            # 如果差异大于0.01才显示precision
            if abs(precision_val - accuracy_val) > 0.01:
                f.write(f"  测试精确率 (Precision): {results['precision']:.6f}\n")
            
            # 删除召回率和F1-score，因为对于手写数字识别任务不是必需的
            
            f.write("\n")
            
            # 写入各类别准确率
            f.write("各类别准确率:\n")
            class_acc = class_wise_accuracy(results)
            for class_name, acc in class_acc.items():
                f.write(f"  数字 {class_name}: {acc:.6f} ({acc*100:.2f}%)\n")
            
            # 写入错误分类统计
            if 'true_labels' in results:
                total_samples = len(results['true_labels'])
                misclassified_indices, _, _ = find_misclassified_samples(results)
                misclassified_count = len(misclassified_indices)
                f.write(f"\n错误分类统计:\n")
                f.write(f"  总样本数: {total_samples}\n")
                f.write(f"  错误分类数: {misclassified_count}\n")
                f.write(f"  错误率: {misclassified_count/total_samples*100:.2f}%\n")
        
        print(f"Test results saved to: {txt_path}")
    
    def run_test(self):
        """运行测试"""
        try:
            self.load_model()
            self.prepare_data()
            results = self.test_model()
            self.analyze_results(results)
            self.generate_visualizations(results)
            
            return results
        except Exception as e:
            print(f"测试过程中出现错误: {e}")
            raise


def main():
    """主函数"""
    args = parse_test_args()
    
    print("MNIST神经网络测试")
    print("=" * 50)
    print(f"模型路径: {args.model_path}")
    print(f"损失函数: {args.loss_function}")
    print(f"批次大小: {args.batch_size}")
    print(f"可视化: 是")
    print(f"保存目录: {args.save_dir}")
    print("=" * 50)
    
    # 创建测试器并运行测试
    tester = ModelTester(args)
    results = tester.run_test()
    
    return results


if __name__ == "__main__":
    main()
