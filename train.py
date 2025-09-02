from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)


def get_train_args(output_dir):
    return TrainingArguments(
        output_dir = output_dir,

        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        
        per_device_train_batch_size=1,
        bf16=True,
        
        warmup_steps=1,
        max_steps=1000,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        save_steps=100,
        save_strategy="steps",
        eval_steps=100,
        eval_strategy="steps",
        do_eval=True,
        gradient_checkpointing=True,
        report_to="none",
        overwrite_output_dir = 'True',
        group_by_length=True,
    )


def train_peft_model(train_dataset, eval_dataset, peft_model, tokenizer):
    peft_model.print_trainable_parameters()

    # Train PEFT adapter
    output_dir = f"./peft-dialogue-summary-training"
    peft_training_args = get_train_args(output_dir)
    
    peft_model.config.use_cache = False
    peft_trainer = Trainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=peft_training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    return peft_trainer.train()