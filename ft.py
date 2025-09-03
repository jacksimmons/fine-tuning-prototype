from huggingface_hub import login
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    set_seed
)
from tqdm import tqdm
from trl import SFTTrainer
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
import torch
from functools import partial
import models, tokenizers, prompt, dataset as ds, train, perf_metrics, eval_model
import testing.zero_shot
from data.gen import gen


DASH_LINE = '-'.join('' for x in range(100))
SEED = 42


# Environment setup
os.environ["WANDB_DISABLED"] = "true"

# Login to HF
login(os.environ.get("HF_TOKEN"))
# Load dataset
dataset = load_dataset("neil-code/dialogsum-test")

# Get base model
model_name = "microsoft/Phi-3-mini-4k-instruct"
model = models.get_model(model_name)
print("Model:")
perf_metrics.print_vram_usage()

tokenizer = tokenizers.get_tokenizer(model_name)

# Test the base model
testing.zero_shot.test_base_model_inference(dataset, models, model_name, SEED)

# Pre-process dataset
max_length = models.get_max_length(model)
print(f"Model max length: {max_length}")
train_dataset = ds.preprocess_dataset(tokenizer, max_length, SEED, dataset["train"])
eval_dataset = ds.preprocess_dataset(tokenizer, max_length, SEED, dataset["validation"])

# Prepare model for QLoRA
model = prepare_model_for_kbit_training(model)

# Setup PEFT for fine-tuning
config = LoraConfig(
    # The rank of the adapter, the lower the fewer parameters you'll need to train
    r=8,
    # Usually 2 * r
    lora_alpha=16,
    target_modules=[
        'o_proj',
        'qkv_proj',
        'gate_up_proj',
        'down_proj'
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

# Enable gradient checkpointing to reduce mem usage
model.gradient_checkpointing_enable()
peft_model = get_peft_model(model, config)
result = train.train_peft_model(train_dataset, eval_dataset, peft_model, tokenizer)
perf_metrics.print_summary(result)

# Load the fine-tuned model from disk
base_model = models.get_model(model_name)
eval_tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    add_bos_token=True,
    trust_remote_code=False,
    use_fast=False
)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

output_dir = f"./peft-dialogue-summary-training"
peft_training_args = train.get_train_args(output_dir)
ft_model = PeftModel.from_pretrained(
    base_model,
    f"{output_dir}/checkpoint-{peft_training_args.max_steps}",
    torch_dtype=torch.float16,
    is_trainable=False
)

# Evaluate the model's trained performance vs untrained
eval_model.qualitative(dataset, ft_model, SEED, gen)
eval_model.quantitative(dataset, model_name, ft_model, gen)

## YODA TUTORIAL REFERENCE
# model_repo = "microsoft/Phi-3-mini-4k-instruct"
# model = get_model(model_repo) 
# print(f"Model:\n{model}")
# print(f"{model.get_memory_footprint()/1e6}MB footprint")

# # Quantised model can be used for inference, but can't be fine-tuned.
# # So we need to setup low-rank adapters (LoRA)
# # LoRA can be attached to each quantised layer (print(model) to see
# # layers)

# # LoRA are mostly regular layers which can be updated as usual, but 
# # they manage to be significantly smaller than the quantised layers.
# # This drastically reduces the number of total trainable params to
# # 1% or less.
# model = prepare_model_for_kbit_training(model)
# config = LoraConfig(
#     # Rank of adapter. Proportional to num params to train.
#     r=8,
#     # Multiplier, usually 2 * r.
#     lora_alpha=16,
#     bias="none",
#     lora_dropout=0.05,
#     task_type="CAUSAL_LM",
#     # Newer models like Phi-3 may require manually setting target
#     # modules
#     target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"]
# )   
# model = get_peft_model(model, config)
# print(f"Peft Model:\n{model}")
# print(f"{model.get_memory_footprint()/1e6}MB footprint")
# train_p, tot_p = model.get_nb_trainable_parameters()
# print(f'Trainable parameters:      {train_p/1e6:.2f}M')
# print(f'Total parameters:          {tot_p/1e6:.2f}M')
# print(f'% of trainable parameters: {100*train_p/tot_p:.2f}%')
# print({p.dtype for p in model.parameters()})

# # Trim dataset
# dataset = load_dataset("dvgodoy/yoda_sentences", split="train")
# dataset = dataset.rename_column("sentence", "prompt")
# dataset = dataset.rename_column("translation_extra", "completion")
# dataset = dataset.remove_columns(["translation"])
# messages = [
#     {"role": "user", "content": dataset[0]['prompt']},
#     {"role": "assistant", "content": dataset[0]['completion']}
# ]

# # Format dataset into conversational format
# def format_dataset(examples):
#     # If prompts is a list (standard)
#     if isinstance(examples["prompt"], list):
#         output_texts = []
#         for i in range(len(examples["prompt"])):
#             converted_sample = [
#                 {"role": "user", "content": examples["prompt"][i]},
#                 {"role": "assistant", "content": examples["completion"][i]}
#             ]
#             output_texts.append(converted_sample)
#         return {"messages": output_texts}
#     else:
#         converted_sample = [
#             {"role": "user", "content": examples["prompt"]},
#             {"role": "assistant", "content": examples["completion"]}
#         ]
#         return {"messages": converted_sample}
# dataset = dataset.map(format_dataset).remove_columns(["prompt", "completion"])

# # Load the tokeniser corresponding to our model.
# tokenizer = get_tokenizer(model_repo)
# print(tokenizer.apply_chat_template(messages, tokenize=False))

# # Fine-tuning - follows exact same training procedure as training from scratch.
# # https://huggingface.co/blog/dvgodoy/fine-tuning-llm-hugging-face#fine-tuning-with-sfttrainer
# sft_config = SFTConfig(
#     # Mem usage
#     gradient_checkpointing=True, # Saves a lot of mem
#     gradient_checkpointing_kwargs={'use_reentrant': False}, # Prevent exceptions in new PyTorch
#     gradient_accumulation_steps=1, # Actual batch (for updating) is same as micro batch size
#     per_device_train_batch_size=16, # Micro batch size to begin with
#     auto_find_batch_size=True, # Halves batch size if it were to cause OOM, until it works

#     # Dataset
#     max_length=64,
#     packing=True, # No padding is needed

#     # Training params
#     num_train_epochs=10,
#     learning_rate=3e-5,
#     # Doesn't help much if using LoRA (which we are)
#     optim="paged_adamw_8bit",

#     # Logging params
#     logging_steps=10,
#     logging_dir="./logs",
#     output_dir="./phi3-mini-yoda-adapter",
#     report_to="none"
# )

# trainer = SFTTrainer(
#     model=model,
#     processing_class=tokenizer,
#     args=sft_config,
#     train_dataset=dataset,
# )

# # Labels were added automatically, and are the same as the inputs
# # Hence "self-supervised fine-tuning"
# trainer.train()
# trainer.save_model("local-phi3-mini-yoda-adapter")

# # Prompt
# prompt = gen_prompt(tokenizer, "The Force is strong with you!")
# print(gen_completion(model, tokenizer, prompt))