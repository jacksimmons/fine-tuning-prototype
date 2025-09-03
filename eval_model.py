# Compare the performance of the original, and trained, models.
from transformers import (
    AutoModelForCausalLM,
    set_seed
)
import numpy as np
import pandas as pd
import evaluate
from models import get_model


DASH_LINE = '-'.join('' for x in range(100))


# Evaluate model qualitatively (human evaluation)
def qualitative(dataset, ft_model, seed, gen):
    set_seed(seed)
    index = 5
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    prompt = f"Instruct: Summarize the following conversation.\n{dialogue}\nOutput:\n"

    peft_model_res = gen(ft_model,prompt,100,)
    peft_model_output = peft_model_res[0].split('Output:\n')[1]
    prefix, success, result = peft_model_output.partition('###')

    print(DASH_LINE)
    print(f'INPUT PROMPT:\n{prompt}')
    print(DASH_LINE)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
    print(DASH_LINE)
    print(f'PEFT MODEL:\n{prefix}')

# Evaluate model quantitatively (ROGUE metric)
# Quantifies the validity of summarisations produced by models
def quantitative(dataset, model_name, ft_model, gen):
    base_model = get_model(model_name)
    dialogues = dataset['test'][0:10]['dialogue']
    human_baseline_summaries = dataset['test'][0:10]['summary']

    original_model_summaries = []
    instruct_model_summaries = []
    peft_model_summaries = []

    for idx, dialogue in enumerate(dialogues):
        human_baseline_text_output = human_baseline_summaries[idx]
        prompt = f"Instruct: Summarize the following conversation.\n{dialogue}\nOutput:\n"
        
        original_model_res = gen(base_model, prompt, 100,)
        original_model_text_output = original_model_res[0].split('Output:\n')[1]
        
        peft_model_res = gen(ft_model, prompt, 100,)
        peft_model_output = peft_model_res[0].split('Output:\n')[1]
        print(peft_model_output)
        peft_model_text_output, success, result = peft_model_output.partition('###')

        original_model_summaries.append(original_model_text_output)
        peft_model_summaries.append(peft_model_text_output)

    zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, peft_model_summaries))

    df = pd.DataFrame(zipped_summaries, columns = ['human_baseline_summaries', 'original_model_summaries', 'peft_model_summaries'])
    rouge = evaluate.load('rouge')

    original_model_results = rouge.compute(
        predictions=original_model_summaries,
        references=human_baseline_summaries[0:len(original_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    peft_model_results = rouge.compute(
        predictions=peft_model_summaries,
        references=human_baseline_summaries[0:len(peft_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    print('ORIGINAL MODEL:')
    print(original_model_results)
    print('PEFT MODEL:')
    print(peft_model_results)

    print("Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL")

    improvement = (np.array(list(peft_model_results.values())) - np.array(list(original_model_results.values())))
    for key, value in zip(peft_model_results.keys(), improvement):
        print(f'{key}: {value*100:.2f}%')