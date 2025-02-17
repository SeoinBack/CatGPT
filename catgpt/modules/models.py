from transformers import (
    GPT2Config, GPT2LMHeadModel,
    BertConfig, BertForSequenceClassification,
    XLNetConfig, XLNetLMHeadModel,
    T5Config, T5ForConditionalGeneration,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForSeq2Seq,
)

from catgpt.modules.t5_modules import DataCollatorForT5MLM, compute_input_and_target_lengths
from catgpt.dataset.dataset import CifDataset

import torch
import numpy as np

def get_model(model_params, data_params, tokenizer):
    
    arch = model_params.architecture
    assert arch in ['GPT','BERT','XLNet','T5']
    
    if arch in ['GPT','BERT','XLNet']:
        if arch == 'GPT':
            data_type = 'cat_txt'
            base_model = GPT2LMHeadModel
            config = GPT2Config(
                vocab_size=len(tokenizer.get_vocab()),
                n_positions=model_params.n_positions,
                n_embd=model_params.n_embd,
                n_layer=model_params.n_layer,
                n_head=model_params.n_head,
            )
             
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
    
              
        elif arch == 'BERT':
            data_type = 'corrupted_cat_txt'
            base_model = BertForSequenceClassification
            config = BertConfig(
                vocab_size=len(tokenizer.get_vocab()),
                max_position_embeddings=model_params.n_positions,
                hidden_size=model_params.n_embd,
                num_hidden_layers=model_params.n_layer,
                num_attention_heads=model_params.n_head,
                num_labels = 1,
            )
            
            data_collator = DataCollatorWithPadding(
                tokenizer=tokenizer,
                padding=True,
            )
            
        elif arch == 'XLNet':
            data_type = 'cat_txt'
            base_model = XLNetLMHeadModel
            config = XLNetConfig(
                vocab_size=len(tokenizer.get_vocab()),
                d_model=model_params.n_embd,
                n_layer=model_params.n_layer,
                n_head=model_params.n_head,
                d_inner=model_params.d_inner,
            )
            
            data_collator = DataCollatorForPermutationLanguageModeling(
                tokenizer=tokenizer,
                plm_probability=model_params.noise_density
            )
        
    elif arch == 'T5':
        data_type = 'cat_txt'
        base_model = T5ForConditionalGeneration
        config = T5Config(
            vocab_size=len(tokenizer.get_vocab()),
            d_model=model_params.n_embd,
            d_kv=(model_params.n_embd // model_params.n_head) if model_params.n_head > 0 else 64,
            d_ff=model_params.n_embd * 4,
            num_layers=model_params.n_layer,
            num_heads=model_params.n_head,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=tokenizer.bos_token_id,
        )
        
        expanded_inputs_length, targets_length = compute_input_and_target_lengths(
            inputs_length=data_params.max_len,
            noise_density=model_params.noise_density,
            mean_noise_span_length=model_params.mean_span,
        )  
        
        data_collator = DataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=model_params.noise_density,
            mean_noise_span_length=model_params.mean_span,
            input_length=data_params.max_len,          
            target_length=targets_length,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=tokenizer.bos_token_id,               
        )
        
        data_params.max_len = expanded_inputs_length
    
    dataset = {
        'train' : CifDataset(
            data_params.train_data_path, 
            tokenizer=tokenizer, 
            data_type=data_type,
            model_type=model_params.architecture,
            string_type=data_params.string_type,
            max_length=data_params.max_len,
            add_props=data_params.add_props,
        ),
        
        'val' : CifDataset(
            data_params.val_data_path, 
            tokenizer=tokenizer,
            data_type=data_type,
            model_type=model_params.architecture,
            string_type=data_params.string_type,
            max_length=data_pars.max_len,
            add_props=data_params.add_props,
        ),
    }
    return base_model, config, dataset, data_collator
