from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class llama1_7b:
    def __init__(self, args):
        print('Loading model llama1-7b...')
        self.tokenizer = LlamaTokenizer.from_pretrained('{model-path}', use_fast=False, padding_side="right")
        self.model = LlamaForCausalLM.from_pretrained('{model-path}', low_cpu_mem_usage = True, torch_dtype=torch.float16)
        
        self.model.cuda()
        
        self.tokenizer.pad_token_id = 0 if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
        self.tokenizer.bos_token_id = 1
        self.model.eval()
        print("Success.\n")


class llama1_13b:
    def __init__(self, args):
        print('Loading model llama1-13b...')
        self.tokenizer = LlamaTokenizer.from_pretrained('{model-path}', use_fast=False, padding_side="right")
        self.model = LlamaForCausalLM.from_pretrained('{model-path}', torch_dtype=torch.float16, device_map='auto')

        self.tokenizer.pad_token_id = 0 if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
        self.tokenizer.bos_token_id = 1
        self.model.eval()
        print("Success.\n")

class llama1_30b:
    def __init__(self, args):
        print('Loading model llama1-30b...')
        self.tokenizer = LlamaTokenizer.from_pretrained('{model-path}', use_fast=False, padding_side="right")
        self.model = LlamaForCausalLM.from_pretrained('{model-path}', torch_dtype=torch.float16, device_map='auto')

        self.tokenizer.pad_token_id = 0 if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
        self.tokenizer.bos_token_id = 1
        self.model.eval()
        print("Success.\n")