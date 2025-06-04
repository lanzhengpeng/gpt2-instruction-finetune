from transformers import BertTokenizer, GPT2LMHeadModel

# 指定模型名称和缓存目录
model_name = "uer/gpt2-chinese-cluecorpussmall"
cache_dir = "./models/gpt2-chinese"

# 下载并保存到本地 cache_dir 目录中
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = GPT2LMHeadModel.from_pretrained(model_name, num_labels=2, cache_dir=cache_dir)

print("模型和分词器加载完成 ✅")
