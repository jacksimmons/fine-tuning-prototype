from transformers import AutoTokenizer


def get_tokenizer(model_name: str):
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=False,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer