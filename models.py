from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM
)
import torch


# BNB setup
def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False
    )


# Gets the LLM, from local filesystem (if present), or downloads it
# from HuggingFace.
def get_model(model_name):
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        device_map="auto",
        quantization_config=get_bnb_config(),
        trust_remote_code=False
    )


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