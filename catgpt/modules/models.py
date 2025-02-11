from transformers import GPT2Config, GPT2LMHeadModel, BertConfig, BertForSequenceClassification, XLNetLMHeadModel, XLNetConfig
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, DataCollatorForPermutationLanguageModeling


def get_model(model_params,tokenizer):

    assert model_params.architecture in ['GPT','BERT','XLNet']
    
    if model_params.architecture == 'GPT':
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
            #return_tensor='pt',
        )

          
    elif model_params.architecture == 'BERT':
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
            #return_tensor='pt',
        )
        
    elif model_params.architecture == 'XLNet':
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
            plm_probability=1.0,
            #max_span_length=5,
        )
    return data_type, base_model, config, data_collator
