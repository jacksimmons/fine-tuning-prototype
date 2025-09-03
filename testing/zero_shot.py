from transformers import (
    AutoTokenizer,
    set_seed
)
from data.gen import gen


DASH_LINE = '-'.join('' for x in range(100))


# Test base model with zero-shot inferencing
def test_base_model_inference(dataset, model, model_name: str, seed):
    eval_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
        add_bos_token=True,
        use_fast=False
    )
    eval_tokenizer.pad_token = eval_tokenizer.eos_token

    set_seed(seed)
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