"""
说明： 数据集已经下载好不用执行此脚本
"""
# 下载数据集
from datasets import load_dataset

# 下载并加载数据集（只运行一次，数据会缓存到 ~/.cache/huggingface）
dataset = load_dataset("BelleGroup/train_1M_CN")

# 将训练集保存到本地 JSON 文件
dataset['train'].to_json("./data/belle_1M_train.json", force_ascii=False)

import json
import random

# 逐行读取 JSONL 文件（每行一个 JSON 对象）
with open("belle_1M_train.json", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# 随机抽取 10000 条
subset = random.sample(data, 10000)

# 保存为标准 JSON 数组格式
with open("./data/belle_10k_random_array.json", "w", encoding="utf-8") as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)

print("✅ 已保存 10000 条随机数据到 ./data/belle_10k_random_array.json")
