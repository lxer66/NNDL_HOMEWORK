#!/bin/bash

# 测试脚本 - 测试models目录下的所有权重文件
# 该脚本会遍历models目录中的所有.npy文件，并对每个文件运行测试

echo "开始测试所有模型权重文件..."
echo "=============================="

# 检查models目录是否存在
if [ ! -d "models" ]; then
    echo "错误: models目录不存在"
    exit 1
fi

# 检查test_results目录，如果不存在则创建
if [ ! -d "test_results" ]; then
    mkdir test_results
    echo "创建test_results目录"
fi

# 计算模型文件总数
total_models=$(ls models/*.npy 2>/dev/null | wc -l)
if [ $total_models -eq 0 ]; then
    echo "警告: models目录中没有找到.npy文件"
    exit 0
fi

echo "找到 $total_models 个模型文件"
echo "=============================="

# 遍历models目录中的所有.npy文件
counter=1
for model_file in models/*.npy; do
    # 提取文件名（不含路径）
    filename=$(basename "$model_file")
    
    echo "[$counter/$total_models] 测试模型: $filename"
    
    # 解析文件名以确定损失函数和正则化方法
    # 文件名格式: optimizer_lossfunction_regularization_final.npy
    IFS='_' read -ra parts <<< "$filename"
    
    # 默认参数
    loss_function="cross_entropy"
    
    # 根据文件名确定损失函数
    if [[ "$filename" == *"mse"* ]]; then
        loss_function="mse"
    elif [[ "$filename" == *"l1_loss"* ]]; then
        loss_function="l1_loss"
    fi
    
    # 运行测试命令
    echo "运行命令: python test.py --model_path $model_file --loss_function $loss_function"
    
    # 执行测试
    python test.py --model_path "$model_file" --loss_function "$loss_function"
    
    if [ $? -eq 0 ]; then
        echo "✓ 测试成功完成"
    else
        echo "✗ 测试失败"
    fi
    
    echo ""
    counter=$((counter + 1))
done

echo "=============================="
echo "所有模型测试完成!"
echo "结果保存在test_results目录中"