import sys
import os

# 获取项目根目录：当前脚本的上上级目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import torch
# from transformers import BertTokenizer, GPT2LMHeadModel
# model = GPT2LMHeadModel.from_pretrained("./saved_model")
# tokenizer = BertTokenizer.from_pretrained("./saved_model")
from transformers import BertTokenizer, GPT2LMHeadModel

model_name = "uer/gpt2-chinese-cluecorpussmall"

tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir="./models/gpt2-chinese")
model = GPT2LMHeadModel.from_pretrained(model_name, num_labels=2, cache_dir="./models/gpt2-chinese")

print("模型和分词器加载完成 ✅")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
import torch
model.to(device)

import json

with open("./data/belle_10k_random_array.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"加载了 {len(data)} 条数据")
print("第一条数据示例:", data[0])

# 数据集划分为训练集、验证集和测试集
train_portion = int(len(data) * 0.85) # 85% for training
test_portion = int(len(data) * 0.1) # 10% for testing
val_portion = len(data) - train_portion - test_portion # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))


from build_dataset import InstructionDataset
from utils.formatting import customized_collate_fn
# 导入 PyTorch 的 DataLoader 类，用于批量加载数据，方便训练和验证。
from torch.utils.data import DataLoader
# 设置 DataLoader 中用于数据加载的子进程数量
num_workers = 0            
# 设置训练/验证/测试时每个 batch 的样本数量为 8
batch_size = 2
# 固定 PyTorch 随机数种子，保证每次运行数据加载顺序和模型初始化等操作是可复现的。
torch.manual_seed(123)
test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

from utils.formatting import format_input
entry = test_data[0]
input_text=format_input(entry)
# 使用分词器讲文本转化成可以处理的张量
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(
    input_ids=inputs["input_ids"].to(device),
    max_new_tokens=500,
    do_sample=False
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))