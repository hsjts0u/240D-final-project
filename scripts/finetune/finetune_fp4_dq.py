# %%
import bitsandbytes as bnb
import copy
from datasets import load_dataset
from dataclasses import dataclass, field
import os
from os.path import exists, join, isdir
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Seq2SeqTrainer,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Sequence, Dict
import torch
from torch.nn.utils.rnn import pad_sequence

# Base model name
base_model_id = "huggyllama/llama-7b"

# %%
def print_trainable_parameters(bits, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

# %%
class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)


# %%
IGNORE_INDEX = -100

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

# %%
# Build tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="right",
    use_fast=False,
    tokenizer_type='llama' if 'llama' in base_model_id else None, # Needed for HF name change
)

_ = tokenizer.add_special_tokens(dict(pad_token=tokenizer._eos_token))

# %%
# Build model
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, # change
        bnb_4bit_quant_type='fp4', # change
    ),
)

# %%
# PEFT
def find_all_linear_names(model):
    targets = [bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear]
    lora_module_names = set()
    for name, module in model.named_modules():
        for cls in targets:
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')

    return list(lora_module_names)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

modules = find_all_linear_names(model)
config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=modules,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

for name, module in model.named_modules():
    if isinstance(module, LoraLayer):
        module = module.to(torch.bfloat16)
    if 'norm' in name:
        module = module.to(torch.float32)
    # if 'lm_head' in name or 'embed_tokens' in name:
    #     if hasattr(module, 'weight'):
    #         if module.weight.dtype == torch.float32:
    #             print(f'{module} : to bfloat16')
    #             module = module.to(torch.bfloat16)

model.config.use_cache = False

# %%
# Prepare training data
data = load_dataset("timdettmers/openassistant-guanaco")
data = data.map(lambda x: {
                    'input': '',
                    'output': x['text'],
                })

data = data.remove_columns(
    [col for col in data.column_names['train'] if col not in ['input', 'output']]
)

train_data = data['train']
train_data = train_data.map(lambda x: {'length': len(x['input']) + len(x['output'])})

# %%
# Build collator
data_collator = DataCollatorForCausalLM(
    tokenizer=tokenizer,
    source_max_len=16,
    target_max_len=512,
    train_on_source=False,
    predict_with_generate=False,
)

# %%
print_trainable_parameters(4, model)

dtypes = {}
for _, p in model.named_parameters():
    dtype = p.dtype
    if dtype not in dtypes: dtypes[dtype] = 0
    dtypes[dtype] += p.numel()
total = 0
for k, v in dtypes.items(): total+= v
for k, v in dtypes.items():
    print(k, v, v/total)

# %%
# Fine-tune
training_args = transformers.TrainingArguments(
    output_dir='/qlora/finetune/dq_fp4', # change
    report_to="none",
    optim="paged_adamw_8bit",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    max_steps=1875,
    weight_decay=0.0,
    learning_rate=0.0002,
    remove_unused_columns=False,
    max_grad_norm=0.3,
    gradient_checkpointing=True,
    lr_scheduler_type='constant',
    warmup_ratio=0.03,
    logging_steps=10,
    group_by_length=True,
    save_strategy='steps',
    save_steps=500,
    save_total_limit=40,
)

training_args.generation_config=transformers.GenerationConfig(
    max_new_tokens=32,
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_data,
    data_collator=data_collator,
)

trainer.add_callback(SavePeftModelCallback)

trainer.train()

# %%



