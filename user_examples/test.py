from roll.pipeline.agentic.env_manager.token_mask_utils import messages_to_tokens_and_masks
import json
import os
from collections import defaultdict
from roll.models.model_providers import default_tokenizer_provider
from roll.datasets.chat_template import get_chat_template
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
def read_jsonl(file_path):
    """
    读取JSONL格式的数据文件。
    
    参数:
        file_path (str): JSONL文件的路径。
    
    返回:
        list[dict]: 包含每一行JSON对象的列表。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每一行的JSON对象
            data.append(json.loads(line.strip()))
    return data

dataset = read_jsonl('/home/chenkaiyuan/evaluation_datasets/tool_sft_processed.jsonl')[:20]
# tokenizer = default_tokenizer_provider('')
tokenizer = AutoTokenizer.from_pretrained(
    '/home/chenkaiyuan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554',
    use_fast=True,
    split_special_tokens=False,
    trust_remote_code=True,
    padding_side="right",
)

tokenized_encodings = []
for line in tqdm(dataset):
    messages = json.loads(line['messages'])
    inputs = tokenizer(line['messages'], return_tensors='pt', truncation=True, max_length=2048)
    input_ids_list, input_ids_list_mask = messages_to_tokens_and_masks(messages, tokenizer)
    assert len(input_ids_list) == len(input_ids_list_mask)
    merged_input_ids, merged_input_ids_mask = [], []
    for input_ids, input_ids_mask in zip(input_ids_list, input_ids_list_mask):
        merged_input_ids.extend(input_ids)
        merged_input_ids_mask.extend(input_ids_mask)
    assert len(merged_input_ids) == len(merged_input_ids_mask)
    label = torch.tensor(merged_input_ids)
    merged_input_ids_mask = torch.tensor(merged_input_ids_mask)
    label[merged_input_ids_mask == 0] = -100
    label = label.tolist()
    attention_mask = [1] * len(merged_input_ids)
    tokenized_encoding = {
        "input_ids": merged_input_ids,
        "attention_mask": attention_mask,
        "labels": label
    }
    tokenized_encodings.append(tokenized_encoding)

res = {key: [tokenized_encoding[key] for tokenized_encoding in tokenized_encodings] for key in tokenized_encodings[0].keys()}
print(res)