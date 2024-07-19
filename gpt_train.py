import torch

from transformers import GPT2Config 
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments

from datasets import load_dataset
from omegaconf import OmegaConf

params = OmegaConf.load('./config/gpt_config.yaml')
model_params = params.model_params
data_params = params.data_params

tokenizer = PreTrainedTokenizerFast.from_pretrained(data_params.tokenizer_path, max_len=data_params.max_len)
#tokenizer.pad_token = tokenizer.eos_token

config = GPT2Config(
    vocab_size=len(tokenizer.get_vocab()),
    n_positions=model_params.n_positions,
    n_embd=model_params.n_embd,
    n_layer=model_params.n_layer,
    n_head=model_params.n_head,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

model = GPT2LMHeadModel(config=config)
datasets = load_dataset('text', data_files={'train':data_params.train_data_path, 'validation':data_params.val_data_path})

def tokenize_function(examples):
    return tokenizer([tokenizer.bos_token + example + tokenizer.eos_token for example in examples["text"]])
    
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

block_size = tokenizer.model_max_length

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
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

#lm_datasets = tokenized_datasets

data_collator = DataCollatorForLanguageModeling(
#data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, mlm=False, #mlm_probability=0.1,
)    
    
training_args = TrainingArguments(
    output_dir='./models/' + model_params.name,
    overwrite_output_dir=True,
    num_train_epochs=model_params.n_epochs,
    per_device_train_batch_size=data_params.batch_size,
    per_device_eval_batch_size=data_params.batch_size,
    save_strategy='epoch',
    #save_steps=5000,   # save checkpoints every 5000 steps
    save_total_limit=5,  # Up to 80 checkpoints can be stored
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
