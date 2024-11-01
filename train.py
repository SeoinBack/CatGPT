import os

from transformers import GPT2Config, GPT2LMHeadModel, BertConfig, BertForSequenceClassification
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model


from catgpt.modules.trainer import CustomHFTrainer
from catgpt.dataset.dataset import CifDataset

from omegaconf import OmegaConf


import wandb
os.environ['WANDB_PROJECT'] ='CatGPT'

params = OmegaConf.load('./config/config.yaml')
model_params = params.model_params
data_params = params.data_params


tokenizer = PreTrainedTokenizerFast.from_pretrained(
    f'./data/tokenizer/{data_params.string_type}-tokenizer/',
    max_len=data_params.max_len
)


if model_params.architecture == 'GPT':
    
    base_model = GPT2LMHeadModel
    
    config = GPT2Config(
        vocab_size=len(tokenizer.get_vocab()),
        n_positions=model_params.n_positions,
        n_embd=model_params.n_embd,
        n_layer=model_params.n_layer,
        n_head=model_params.n_head,
    )
elif model_params.architecture == 'BERT':
    base_model = BertForSequenceClassification
    
    config = BertConfig(
        vocab_size=len(tokenizer.get_vocab()),
        max_position_embeddings=model_params.n_positions,
        hidden_size=model_params.n_embd,
        num_hidden_layers=model_params.n_layer,
        num_attention_heads=model_params.n_head,
        num_labels = 1,
    )

if model_params.use_pretrained:
    model = base_model.from_pretrained(
        pretrained_model_name_or_path = model_params.checkpoint_path
        )
    
    lora_config = LoraConfig(
        r = model_params.r,
        lora_alpha = model_params.lora_alpha,
        lora_dropout = model_params.lora_dropout,
        task_type = model_params.task_type
    )
    
    if use_lora:
        model = get_peft_model(model, lora_config)
    
else:
    model =  base_model(config=config)
    

dataset = {
    'train' : CifDataset(data_params.train_data_path, tokenizer, string_type=data_params.string_type),
    'val' : CifDataset(data_params.val_data_path, tokenizer, string_type=data_params.string_type)
}

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    #return_tensor='pt',
)

training_args = TrainingArguments(
    output_dir = f'./outputs/{model_params.name}/',
    overwrite_output_dir = True,
    do_train = True,
    do_eval = True,
    eval_strategy   = 'steps',
    eval_steps = 5000,
    per_device_train_batch_size = data_params.batch_size,
    per_device_eval_batch_size = data_params.batch_size,
    gradient_accumulation_steps = data_params.gradient_accumulation_steps,
    save_strategy = 'epoch',
    run_name = model_params.name,
    report_to = 'wandb',
    num_train_epochs=15,
    save_total_limit =5,
)

if model_params.use_numerical_encoding:
    trainer = CustomHFTrainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = dataset['train'],
        eval_dataset = dataset['val'],
        tokenizer = tokenizer,
        use_numerical_encodings=True,
        d_model = model_params.n_embd,
        vocab_size = len(tokenizer.get_vocab()),
    )
    
else:
    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = dataset['train'],
        eval_dataset = dataset['val'],
        tokenizer = tokenizer,
    )
    
trainer.train()
trainer.save_model('./outputs/' + f'{model_params.name}/')
