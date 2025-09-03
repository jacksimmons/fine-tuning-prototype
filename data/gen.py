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