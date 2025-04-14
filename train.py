import os

from transformers import PreTrainedTokenizerFast
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

from catgpt.modules.trainer import CustomHFTrainer
from catgpt.modules.models import get_model
from catgpt.modules.tokenizers import T5TokenizerForCat
from catgpt.dataset.dataset import CifDataset

from omegaconf import OmegaConf


import wandb
os.environ['WANDB_PROJECT'] ='CatGPT'

params = OmegaConf.load('./config/config.yml')
model_params = params.model_params
data_params = params.data_params

if data_params.add_props:
    props = 'prop-'
else:
    props = ''

if data_params.string_type == 't5':
    tokenizer = T5TokenizerForCat.from_pretrained(
            f'./data/tokenizer/t5-{props}tokenizer/',
    )

else:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        f'./data/tokenizer/{data_params.string_type}-{props}tokenizer/',
        max_len=data_params.max_len
    )



base_model, config, dataset, data_collator = get_model(model_params, data_params, tokenizer)


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
    
    if model_params.use_lora:
        model = get_peft_model(model, lora_config)
    
else:
    model =  base_model(config=config)

training_args = TrainingArguments(
    output_dir = f'./outputs/{model_params.name}/',
    overwrite_output_dir = True,
    do_train = True,
    do_eval = True,
    eval_strategy   = 'steps',
    eval_steps = data_params.eval_steps,
    per_device_train_batch_size = data_params.batch_size,
    per_device_eval_batch_size = data_params.batch_size,
    gradient_accumulation_steps = data_params.gradient_accumulation_steps,
    save_strategy = 'epoch',
    run_name = model_params.name,
    report_to = 'wandb',
    num_train_epochs=data_params.num_epochs,
    save_total_limit =5,
    learning_rate=data_params.learning_rate,
    warmup_steps=data_params.warmup_steps,
    dataloader_num_workers=data_params.num_workers,
    tf32 = data_params.tf32,
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
