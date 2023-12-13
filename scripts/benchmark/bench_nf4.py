import os
from os.path import exists, join, isdir
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer


# TODO: Update variables
max_new_tokens = 64
top_p = 0.9
temperature=0.7
user_question = "What is Einstein's theory of relativity?"

# Base model
model_name_or_path = 'huggyllama/llama-7b'
# Adapter name on HF hub or local checkpoint path.
adapter_path = '/qlora/finetune/nf4/checkpoint-1875/adapter_model'
# adapter_path = 'timdettmers/guanaco-7b'

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Load the model (use bf16 for faster inference)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    load_in_4bit=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type='nf4',
    )
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

print(model.get_memory_footprint())

prompt = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "### Human: {user_question}"
    "### Assistant: "
)

def generate(model, user_question, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    inputs = tokenizer(prompt.format(user_question=user_question), return_tensors="pt").to('cuda')

    start = time.time()
    outputs = model.generate(
        **inputs, 
        generation_config=GenerationConfig(
            do_sample=True,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
    )
    end = time.time()

    print(f'tok/s: {len(outputs[0]) / (end-start)}')
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)
    return text

generate(model, user_question)