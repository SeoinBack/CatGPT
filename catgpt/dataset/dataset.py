import pandas as pd

from torch.utils.data import Dataset
from catgpt.dataset.dataset_utils import str_preprocess, prop_preprocess

class CifDataset(Dataset):
    '''
    Define custom dataset class for crystal structure generation
    
    todo: add compatibility check between string type and tokenizer
    '''
    
    def __init__(
        self,
        csv_fn,
        tokenizer=None,
        data_type='cat_txt',
        model_type='GPT',
        string_type='coordinate',
        max_length=1024,
        add_props=False,
        augment_type=None,
    ):
        super().__init__()

        df = pd.read_csv(csv_fn)
        self.inputs = df.to_dict(orient='records')
        self.tokenizer = tokenizer
        self.data_type = data_type
        self.model_type = model_type
        self.string_type = string_type
        self.augment_type = augment_type
        self.max_length = max_length
        self.add_props = add_props

    def get_value_from_key(self, input_dict, key):
        return input_dict[key]

    def tokenize(self, input_dict):
        input_str = self.get_value_from_key(input_dict, self.data_type)
        
        #if self.string_type == 'ads':
        #    ads = self.get_value_from_key(input_dict, 'ads_symbol')
        #else:
        #    ads = None
        
        input_str = str_preprocess(
                string_type=self.string_type, 
                input_str=input_str, 
                augment_type=self.augment_type,
        #        ads=ads
                )
        
        if self.add_props:
            prop_str = prop_preprocess(
                input_dict
                )
            input_str = ' '.join([prop_str,input_str])
        
        # tokenize crystal strings with bos and eos token
        input_tokens = self.tokenizer(
            ' '.join([self.tokenizer.bos_token, input_str, '.', self.tokenizer.eos_token]),
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            #add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )
        
        attention_mask = input_tokens.attention_mask[0]
        input_ids = input_tokens.input_ids[0]
        
        if self.model_type in ['GPT','XLNet']:
            labels = input_ids
            
            input_dict = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
        elif self.model_type == 'BERT':
            labels = self.get_value_from_key(input_dict, 'corruption_label')
            
            input_dict = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        
        elif self.model_type in ['T5','BART']:
            input_dict = dict(
                input_ids=input_ids,
                #attention_mask=attention_mask,
            )
        
        return input_dict
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")
        vals = self.inputs[index]
        vals = self.tokenize(vals)
        return vals