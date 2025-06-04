import json


# 导入 PyTorch 及其 Dataset 类，用于构建自定义数据集。
import torch
from torch.utils.data import Dataset
# 导入format_input
from utils.formatting import format_input
class InstructionDataset(Dataset):
    # data：原始数据列表，每条数据是一个字典，包含 instruction、input、output 等字段
    # tokenizer：分词器，用于将文本转换成模型可读的 token id。
    def __init__(self, data, tokenizer):
        # 将传入的数据保存为实例属性 self.data，方便后续使用
        self.data = data
        # 创建一个空列表 self.encoded_texts，用来存放编码后的文本（token id 序列）
        self.encoded_texts = []
        # 遍历数据中的每一条记录 entry，准备对每条数据进行处理和编码
        for entry in data:
            # 遍历数据中的每一条记录 entry，准备对每条数据进行处理和编码。                                           
            instruction_plus_input = format_input(entry)
            # 拼接响应部分文本，格式是两行换行 + ### 响应: 标记 + 输出内容，形成完整的指令-输入-输出结构。
            response_text = f"\n\n### 响应:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            # 使用传入的分词器对 full_text 进行编码（转换成 token id 列表），并把编码结果追加到 self.encoded_texts 列表中。
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
    # 定义获取数据项的方法，支持通过索引获取编码后的文本
    def __getitem__(self, index):
        return self.encoded_texts[index]
    # 定义返回数据集长度的方法。
    def __len__(self):
        return len(self.data)



