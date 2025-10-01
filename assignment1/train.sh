#!/bin/bash

# 训练脚本：执行20次训练实验
# 1. 固定优化器为sgd，组合不同损失函数和正则化方法（包括none）：3*5=15次
# 2. 固定损失函数为cross_entropy，正则化方法为l2，使用除了sgd以外的其他优化器：1*1*4=4次
# 3. 固定损失函数为cross_entropy，优化器为sgd，正则化方法为none，使用不同学习率：1次

echo "开始执行训练实验..."

# 第一部分：固定优化器为sgd，组合不同损失函数和正则化方法（包括none）
echo "第一部分：固定优化器为sgd，组合不同损失函数和正则化方法"

echo "实验1: sgd + cross_entropy + none"
python train.py --optimizer sgd --loss cross_entropy --regularization none --epochs 100

echo "实验2: sgd + cross_entropy + l1"
python train.py --optimizer sgd --loss cross_entropy --regularization l1 --lambda_reg 0.0001 --epochs 100

echo "实验3: sgd + cross_entropy + l2"
python train.py --optimizer sgd --loss cross_entropy --regularization l2 --lambda_reg 0.001 --epochs 100

echo "实验4: sgd + cross_entropy + dropout"
python train.py --optimizer sgd --loss cross_entropy --regularization dropout --dropout_rate 0.1 --epochs 100

echo "实验5: sgd + cross_entropy + data_augmentation"
python train.py --optimizer sgd --loss cross_entropy --regularization data_augmentation --epochs 100

echo "实验6: sgd + mse + none"
python train.py --optimizer sgd --loss mse --regularization none --epochs 100

echo "实验7: sgd + mse + l1"
python train.py --optimizer sgd --loss mse --regularization l1 --lambda_reg 0.0001 --epochs 100

echo "实验8: sgd + mse + l2"
python train.py --optimizer sgd --loss mse --regularization l2 --lambda_reg 0.001 --epochs 100

echo "实验9: sgd + mse + dropout"
python train.py --optimizer sgd --loss mse --regularization dropout --dropout_rate 0.1 --epochs 100

echo "实验10: sgd + mse + data_augmentation"
python train.py --optimizer sgd --loss mse --regularization data_augmentation --epochs 100

echo "实验11: sgd + l1_loss + none"
python train.py --optimizer sgd --loss l1_loss --regularization none --epochs 100

echo "实验12: sgd + l1_loss + l1"
python train.py --optimizer sgd --loss l1_loss --regularization l1 --lambda_reg 0.0001 --epochs 100

echo "实验13: sgd + l1_loss + l2"
python train.py --optimizer sgd --loss l1_loss --regularization l2 --lambda_reg 0.001 --epochs 100

echo "实验14: sgd + l1_loss + dropout"
python train.py --optimizer sgd --loss l1_loss --regularization dropout --dropout_rate 0.1 --epochs 100

echo "实验15: sgd + l1_loss + data_augmentation"
python train.py --optimizer sgd --loss l1_loss --regularization data_augmentation --epochs 100

# 第二部分：固定损失函数为cross_entropy，正则化方法为dropout，使用除了sgd以外的其他优化器
echo "第二部分：固定损失函数为cross_entropy，正则化方法为dropout，使用除了sgd以外的其他优化器"

echo "实验16: momentum + cross_entropy + dropout"
python train.py --optimizer momentum --loss cross_entropy --regularization dropout --epochs 100

echo "实验17: adam + cross_entropy + dropout"
python train.py --optimizer adam --loss cross_entropy --regularization dropout --epochs 100

echo "实验18: adamw + cross_entropy + dropout"
python train.py --optimizer adamw --loss cross_entropy --regularization dropout --epochs 100 --weight_decay 0.001

echo "实验19: adagrad + cross_entropy + dropout"
python train.py --optimizer adagrad --loss cross_entropy --regularization dropout --epochs 100

echo "所有实验训练完成！"