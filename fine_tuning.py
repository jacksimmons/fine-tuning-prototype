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
from model import get_bnb_config, get_model
from perf_metrics import print_vram_usage, print_summary
from train import get_train_args, train_peft_model
from eval_model import qualitative, quantitative


DASH_LINE = '-'.join('' for x in range(100))
SEED = 42


# Environment setup
os.environ["WANDB_DISABLED"] = "true"

# Login to HF
login(os.environ.get("HF_TOKEN"))

# Load dataset
dataset = load_dataset("neil-code/dialogsum-test")

# Get untrained model
model_name = "microsoft/Phi-3-mini-4k-instruct"
model = get_model(model_name)
print("Model:")
print_vram_usage()

# Setup tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_name,
    trust_remote_code=True,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
    use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token

# Test base model with zero-shot inferencing
eval_tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    add_bos_token=True,
    use_fast=False
)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

def gen(model, p, maxLen=100, sample=True):
    toks = eval_tokenizer(p, return_tensors="pt")
    res = model.generate(
        **toks.to("cuda"),
        max_new_tokens=maxLen,
        do_sample=sample,
        num_beams=1,
        top_p=0.95
    ).to("cpu")
    return eval_tokenizer.batch_decode(res, skip_special_tokens=True)

set_seed(SEED)
index = 10
prompt = dataset["test"][index]["dialogue"]
summary = dataset["test"][index]["summary"]
formatted_prompt = f"Instruct: Summarize the following conversation.\n{prompt}\nOutput:\n"
res = gen(model, formatted_prompt, 100,)
output = res[0].split("Output:\n")[1]

print(DASH_LINE)
print(f"INPUT PROMPT:\n{formatted_prompt}")
print(DASH_LINE)
print(f"BASELINE HUMAN SUMMARY:\n{summary}\n")
print(DASH_LINE)
print(f"MODEL GENERATION - ZERO SHOT:\n{output}")

# Helper functions to format our input dataset for fine-tuning
def create_prompt_formats(sample):
    # Format fields of the sample ("instruction", "output"), concatenate them
    # using two newlines.
    INTRO_BLURB = "Below is an instruction describing a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruct: Summarize the below conversation."
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"

    blurb = f"\n{INTRO_BLURB}"
    instruction = f"\n{INSTRUCTION_KEY}"
    input_context = f"{sample["dialogue"]}" if sample["dialogue"] else None
    response = f"{RESPONSE_KEY}\n{sample["summary"]}"
    end = f"{END_KEY}"

    parts = [blurb, instruction, input_context, response, end]
    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt

    return sample
# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(conf, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length
def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )
# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset):
    """
    Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """
    
    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)#, batched=True)
    
    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=['id', 'topic', 'dialogue', 'summary'],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

# Pre-process dataset
max_length = get_max_length(model)
print(f"Model max length: {max_length}")
train_dataset = preprocess_dataset(tokenizer, max_length, SEED, dataset["train"])
eval_dataset = preprocess_dataset(tokenizer, max_length, SEED, dataset["validation"])

# Prepare model for QLoRA
model = prepare_model_for_kbit_training(model)

# Setup PEFT for fine-tuning
config = LoraConfig(
    r=32, #Rank
    lora_alpha=32,
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'dense'
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

# Enable gradient checkpointing to reduce mem usage
model.gradient_checkpointing_enable()
peft_model = get_peft_model(model, config)
result = train_peft_model(train_dataset, eval_dataset, peft_model, tokenizer)
print_summary(result)

# Load the fine-tuned model from disk
base_model = get_model(model_name)
eval_tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    add_bos_token=True,
    trust_remote_code=True,
    use_fast=False
)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

output_dir = f"./peft-dialogue-summary-training"
peft_training_args = get_train_args(output_dir)
ft_model = PeftModel.from_pretrained(
    base_model,
    f"{output_dir}/checkpoint-{peft_training_args.max_steps}",
    torch_dtype=torch.float16,
    is_trainable=False
)

# Evaluate the model
qualitative(dataset, ft_model, SEED, gen)
quantitative(dataset, model_name, get_bnb_config(), ft_model, gen)

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