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
        trust_remote_code=True
    )