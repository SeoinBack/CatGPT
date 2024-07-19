import torch

from transformers import GPT2Config 
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments

from datasets import load_dataset
from omegaconf import OmegaConf

from peft import LoraConfig, get_peft_model


params = OmegaConf.load('./config/gpt_finetune_config.yaml')
model_params = params.model_params
data_params = params.data_params

tokenizer = PreTrainedTokenizerFast.from_pretrained(data_params.tokenizer_path, max_len=data_params.max_len)

model = GPT2LMHeadModel.from_pretrained(
    pretrained_model_name_or_path = model_params.pretrained_path
    )

lora_config = LoraConfig(
    r = model_params.lora_rank,
    lora_alpha = model_params.lora_alpha,
    lora_dropout = model_params.lora_dropout,
    task_type = 'CAUSAL_LM'
)

#model = get_peft_model(model, lora_config)
#model.print_trainable_parameters()

datasets = load_dataset('text', data_files={'train':data_params.train_data_path, 'validation':data_params.val_data_path})

def tokenize_function(examples):
    return tokenizer([tokenizer.bos_token + example + tokenizer.eos_token for example in examples["text"]])
    
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

block_size = tokenizer.model_max_length

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)  

training_args = TrainingArguments(
    output_dir='./models/' + model_params.name,
    overwrite_output_dir=True,
    num_train_epochs=model_params.n_epochs,
    per_device_train_batch_size=data_params.batch_size,
    per_device_eval_batch_size=data_params.batch_size,
    save_steps=5000,   # save checkpoints every 5000 steps
    save_total_limit=10,  # Up to 80 checkpoints can be stored
    do_train=True,
    do_eval=True,
    evaluation_strategy='steps',
    prediction_loss_only=True,
    gradient_accumulation_steps=data_params.gradient_accumulation_steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator
)
trainer.train()
trainer.save_model('./models/' + model_params.name)
