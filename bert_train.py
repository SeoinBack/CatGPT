from transformers import BertForSequenceClassification, PreTrainedTokenizerFast, DataCollatorWithPadding
from transformers import BertConfig, TrainingArguments, Trainer
from datasets import load_dataset
from omegaconf import OmegaConf

import pickle
import evaluate
import numpy as np

params = OmegaConf.load('./config/bert_config.yaml')
model_params = params.model_params
data_params = params.data_params

tokenizer =PreTrainedTokenizerFast.from_pretrained(data_params.tokenizer_path, max_len=data_params.max_len)
#tokenizer.pad_token = tokenizer.eos_token

bert_config = BertConfig(
    vocab_size=len(tokenizer.get_vocab()),
    max_position_embeddings=data_params.max_len,
    hidden_size=model_params.hidden_size,
    num_hidden_layers=model_params.num_hidden_layers,
    num_attention_heads=model_params.num_attention_heads,
    pad_token_id=tokenizer.pad_token_id,
    num_labels = 1,
)

if model_params.from_pretrained == None:
    model = BertForSequenceClassification(bert_config)
else:
    model = BertForSequenceClassification.from_pretrained(model_params.from_pretrained)

accuracy = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
    
datasets = load_dataset('text', data_files={'train':data_params.data_path + '/train.txt', 'validation':data_params.data_path + '/val.txt'})

with open(data_params.data_path+'/train_label.pkl','rb') as fr:
    train_labels = pickle.load(fr)
with open(data_params.data_path+'/val_label.pkl','rb') as fr:
    val_labels = pickle.load(fr)
    
datasets['train'] = datasets['train'].add_column('labels',train_labels)
datasets['validation'] = datasets['validation'].add_column('labels',val_labels)

def tokenize_function(examples):
    return tokenizer([example.split(' . ')[0] for example in examples["text"]])

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
)  

training_args = TrainingArguments(
    output_dir="./models/" + model_params.name,
    run_name=model_params.name,
    per_device_train_batch_size=data_params.batch_size,
    per_device_eval_batch_size=data_params.batch_size,
    num_train_epochs=model_params.n_epochs,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    save_strategy='epoch',
    #save_steps=5000,
    save_total_limit=10,
    #load_best_model_at_end=True,
    push_to_hub=False,
    #prediction_loss_only=True,
    gradient_accumulation_steps=data_params.gradient_accumulation_steps,
    learning_rate=1e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model('./models/' + model_params.name)
