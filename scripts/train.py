import sys
import os

# 获取项目根目录：当前脚本的上上级目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

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
import torch
from transformers import BertTokenizer, GPT2LMHeadModel
from utils.formatting import customized_collate_fn
# 加载模型和分词器
model_name = "uer/gpt2-chinese-cluecorpussmall"

tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir="./models/gpt2-chinese")
model = GPT2LMHeadModel.from_pretrained(model_name, num_labels=2, cache_dir="./models/gpt2-chinese")

print("模型和分词器加载完成 ✅")
# 导入 PyTorch 的 DataLoader 类，用于批量加载数据，方便训练和验证。
from torch.utils.data import DataLoader
# 设置 DataLoader 中用于数据加载的子进程数量
num_workers = 0            
# 设置训练/验证/测试时每个 batch 的样本数量为 8
batch_size = 2
# 固定 PyTorch 随机数种子，保证每次运行数据加载顺序和模型初始化等操作是可复现的。
torch.manual_seed(123)
# 用训练数据 train_data 和分词器 tokenizer 实例化自定义数据集 InstructionDataset。
train_dataset = InstructionDataset(train_data, tokenizer)
# 以 batch_size=8 进行批处理
# 使用你自定义的 customized_collate_fn 函数对批次数据进行整理和填充
# shuffle=True 表示每个 epoch 打乱数据顺序，有助于训练效果
# drop_last=True 表示如果最后一个批次数据不足8条，则丢弃，保证每个 batch 大小一致
# 使用前面定义的 num_workers=0 控制加载进程数
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)
# 讲模型放入指定设备中确保和数据加载放入的相同
from utils.formatting import device
model = model.to(device)


# 开始训练
from utils.formatting import train_model_simple,format_input
# 导入 time 模块，用于测量训练过程耗时
import time
# 记录训练开始时间（单位是秒），用于后面计算总耗时。
start_time = time.time()
# 设置 PyTorch 的随机种子为 123，以确保结果可复现（例如初始化、打乱顺序等都是一样的）
torch.manual_seed(123)
# model.parameters()：模型的所有参数；
# lr=0.00005：学习率；
# weight_decay=0.1：L2 正则项，防止过拟合。
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
# 指定训练轮数为 2。即整个数据集会被训练两遍。
num_epochs = 2
# 调用你定义好的 train_model_simple() 训练函数，开始模型微调
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")


# 打印视图
import matplotlib.pyplot as plt
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny() #A
    ax2.plot(tokens_seen, train_losses, alpha=0) #B
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

#A 创建与 y 轴共用的第二个 x 轴
#B 用于对齐刻度的隐藏图形

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# 保存权重
save_dir = "./saved_model"

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"模型和分词器已保存到 {save_dir} 文件夹")
