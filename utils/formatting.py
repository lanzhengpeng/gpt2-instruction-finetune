"""
数据集处理为指令
"""
# format_input函数用于将数据列表中的条目转换Alpaca 风格输入格式
def format_input(entry):
    instruction_text = (f"下面是描述任务的指令。"
                        f"编写一个适当地完成请求的响应。"
                        f"\n\n### 指令:\n{entry['instruction']}")
    input_text = f"\n\n### 输入:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


"""
数据处理函数
"""
import torch
# batch: 当前批次中的样本（每个是一个 token id 的列表）
# pad_token_id=102: 用于填充序列的 token id
# ignore_index=-100: 在计算损失（如 CrossEntropy）时，忽略该索引。
# allowed_max_length: 可选参数，限制每条序列的最大长度（用于截断）
def custom_collate_fn(batch,pad_token_id=102,ignore_index=-100,allowed_max_length=None,device="cpu"):
    # 找出批次中最长序列的长度，加 1 是为了添加 pad
    batch_max_length = max(len(item)+1 for item in batch)
    # 初始化输入和标签列表
    inputs_lst, targets_lst = [], []
    # 遍历每个样本，做处理
    for item in batch:
        # 复制一个样本 item，防止对原始数据造成修改。
        new_item = item.copy()
        # 添加一个 pad 或 eos token，防止后移时丢掉末尾信息
        new_item += [pad_token_id]
        # 补齐 padding，让该样本长度达到 batch_max_length。
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        # inputs: 去掉最后一个 token，作为模型输入
        inputs = torch.tensor(padded[:-1]) 
        # targets: 去掉第一个 token，作为预测目标
        targets = torch.tensor(padded[1:]) 
        # mask: 找到 targets 中所有 pad 的位置。
        mask = targets == pad_token_id                  
        # indices: 是所有 pad 出现的位置索引。
        indices = torch.nonzero(mask).squeeze()         
        # if indices.numel() > 1: 如果有多个 pad，只保留第一个 pad，其余都设为 ignore_index
        if indices.numel() > 1:                         
            targets[indices[1:]] = ignore_index         
        # 如果设定了最大长度（如 allowed_max_length=512），则对 inputs 和 targets 进行截断
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]        
            targets = targets[:allowed_max_length]      
        # 将每个样本的输入和标签加入列表。
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    # 堆叠为张量并返回
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst)
    return inputs_tensor, targets_tensor

# 再包装一次函数
from functools import partial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
customized_collate_fn = partial(custom_collate_fn, device=device,
allowed_max_length=1024)



"""
计算一个批次的损失
"""
# 定义一个函数 calc_loss_batch，计算一个 batch 的损失值。
# input_batch：输入数据的 batch（通常是 token id 矩阵，形状如 [batch_size, seq_len]）
# target_batch：目标标签 batch（通常是对应的 token id）
# model：模型实例
# device：设备（CPU 或 GPU），比如 "cuda" 或 "cpu"
def calc_loss_batch(input_batch, target_batch, model, device):
    # 调用 .to(device) 把输入数据和目标数据都放到同一个设备上（一般是 GPU）。
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  
    # 把 input_batch 输入到模型，得到预测的输出 logits。     
    logits = model(input_batch).logits
    # 计算预测结果与目标标签的交叉熵损失
    # logits.flatten(0, 1)：将 logits 的第 0 维（batch）和第 1 维（序列长度）合并成一个维度，变成 [batch_size * seq_len, vocab_size]，方便计算每个 token 的分类损失。
    # target_batch.flatten()：将目标标签也展平为 [batch_size * seq_len] 的一维向量，和预测一一对应。
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

"""
计算一个数据加载器（data_loader）中若干个批次的平均损失
"""
# 定义函数 calc_loss_loader，计算一个数据加载器（data_loader）中若干个批次的平均损失
def calc_loss_loader(data_loader, model, device, num_batches=None):
    # 初始化变量 total_loss，用来累积所有批次的损失，初始为0.0（浮点数）。
    total_loss = 0.
    # 判断 data_loader 是否为空（长度为0）。
    # 如果没有数据批次，则返回 nan（表示“不是一个数字”），避免后续计算错误。
    if len(data_loader) == 0:
        return float("nan")
    # 如果 num_batches 参数未传入（为 None），则默认计算全部批次，即用 data_loader 的长度作为批次数。
    elif num_batches is None:
        num_batches = len(data_loader)                                    
    else:
        # 如果传入了 num_batches，表示只想计算前多少批次的损失。
        # 这行代码确保 num_batches 不超过 data_loader 的总批次数，取二者中较小值，避免越界。
        num_batches = min(num_batches, len(data_loader))  
    # 遍历 data_loader，i 是批次索引，input_batch 和 target_batch 是每个批次的输入和目标数据                
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # 调用前面你给的函数 calc_loss_batch，计算当前批次的损失。
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 将本批次的损失数值（用 .item() 把 tensor 转成纯 Python 数字）累加到 total_loss。
            total_loss += loss.item()                                     
        else:
            break
    return total_loss / num_batches  

"""
评估模型 训练集和验证集上的损失。
"""
# 定义函数 evaluate_model，用于评估模型在训练集和验证集上的损失。
# model：待评估的模型
# train_loader：训练集的数据加载器。
# val_loader：验证集的数据加载器。
# device：计算设备（CPU或GPU）。
# eval_iter：指定评估时最多计算多少个批次。
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # 换模型到评估模式（eval mode）。
    model.eval()      
    # 进入一个不计算梯度的上下文环境。          
    with torch.no_grad(): 
        # 使用前面定义的 calc_loss_loader 函数，计算训练集上 eval_iter 个批次的平均损失。      
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        # 同样计算验证集上 eval_iter 个批次的平均损失。
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    # 评估完毕，恢复模型为训练模式（train mode）。
    model.train()
    # 返回训练集和验证集上的平均损失，方便观察模型性能
    return train_loss, val_loss

"""
生成样本函数
"""
# 定义函数，基于给定的模型和起始文本生成示例文本并打印出来。
# model：预训练或正在训练的大语言模型
# tokenizer：用于文本编码和解码的分词器。
# device：模型和数据运行的设备（CPU或GPU）
# start_context：生成文本的起始上下文字符串。
def generate_and_print_sample(model, tokenizer, device, start_context):
    # 切换模型到评估模式（eval mode），禁用 dropout 等随机行为，保证生成稳定。
    model.eval()
    # 读取模型的最大上下文长度（即位置编码的长度），用于限制生成时输入上下文的最大长度。
    # context_size = model.pos_emb.weight.shape[0]
    # 使用分词器将起始上下文 start_context 编码成 token ID 序列
    # 调用自定义函数 text_to_token_ids（假设是把文本转成模型输入的数字ID）。
    encoded = tokenizer(start_context, return_tensors="pt").to(device)
    # 进入无梯度计算上下文，不计算梯度，节省内存和加快推理速度。
    with torch.no_grad():
        # 调用 generate_text_simple 函数，让模型基于 encoded 起始 token，生成最多 50 个新的 token。
        token_ids = model.generate(
                    input_ids=encoded["input_ids"],
                    max_new_tokens=500,   # ✅ 生成最多500个新 token
                    do_sample=False       # ✅ 不使用采样，改为贪心或束搜索
                    )
        # 调用 token_ids_to_text，将生成的 token ID 序列解码回可读文本字符串。
        decoded_text=tokenizer.decode(token_ids[0], skip_special_tokens=True)
        # 打印生成文本，将换行符替换为空格，使输出在终端显示更紧凑、整齐。
        print(decoded_text.replace("\n", " ")) 
    # 生成结束后，恢复模型为训练模式（train mode），准备继续训练。
    model.train()

"""
模型训练函数
"""
# 定义一个简单的训练函数，用于大语言模型（LLM）的预训练。
# model你要训练的大语言模型（LLM）
# train_loader训练集的数据加载器。
# val_loader验证集的数据加载器
# optimizer 优化器对象（如 Adam、SGD）
# device 计算设备，通常是 "cpu" 或 "cuda"（GPU）
# num_epochs 训练的总轮数（epoch 数），即数据集整体遍历次数。
# eval_freq评估频率，单位是训练步数（batch 数）。例如 eval_freq=100，表示每训练100步进行一次模型评估
# eval_iter评估时，计算损失时最多使用多少个批次的数据。控制评估开销，比如用前100个batch计算平均损失。
# start_context用于生成文本的初始上下文字符串。在每个 epoch 结束时，模型基于这个上下文生成示例文本，用于观察训练效果。
# tokenizer分词器对象，负责将文本转成模型能处理的 token ID，以及将模型输出的 token ID 转回文本。用于生成示例文本时解码。
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # train_losses：记录每次评估时训练集的平均损失。
    # val_losses：记录验证集的平均损失。
    # track_tokens_seen：记录训练过程中已处理的 token 数量。
    train_losses, val_losses, track_tokens_seen = [], [], []  
    # tokens_seen：累计已处理的 token 数量，用于监控训练进度。
    # global_step：全局步数计数器，初始化为 -1（下一步为0）。
    tokens_seen, global_step = 0, -1
    # 进入主训练循环，迭代指定的训练轮数（epoch）。
    for epoch in range(num_epochs):    
        # 将模型切换为训练模式，启用 dropout 和 batchnorm 的训练状态。                                             
        model.train()
        # 遍历训练数据加载器中的每个批次，取得输入和目标标签。
        for input_batch, target_batch in train_loader:
            # .to(device) 就能让训练跑在 GPU 上：
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            # 清零上一个批次的梯度，避免梯度累积。
            optimizer.zero_grad()   
            # 调用前面定义的函数计算当前批次的损失                                                
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 对损失进行反向传播，计算各参数的梯度。
            loss.backward()    
            # 使用优化器根据计算的梯度更新模型权重。                                                     
            optimizer.step()   
            # 累加当前批次中所有元素（token）的数量到 tokens_seen。                                                     
            tokens_seen += input_batch.numel()
            # 全局训练步数加一。
            global_step += 1
            # 每训练 eval_freq 步，执行一次评估。
            if global_step % eval_freq == 0: 
                # 调用评估函数，计算训练集和验证集的平均损失                                       
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                # 将这次评估得到的训练损失、验证损失和已处理 token 数保存到对应列表中。
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # 打印当前 epoch、训练步数和对应的训练/验证损失，方便观察训练过程。
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        # 每个 epoch 结束后，调用生成函数基于 start_context 生成并打印示例文本，直观展示模型当前语言生成能力。
        generate_and_print_sample(                                                  
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen
