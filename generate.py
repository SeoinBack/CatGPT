import pickle
import torch

from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from omegaconf import OmegaConf
from tqdm import tqdm

params = OmegaConf.load('./config/gen_config.yaml').params

print(f'Load input from {params.input_path}...')
with open(params.input_path,'rb') as fr:
    input_str_list = pickle.load(fr)

print(f'Load tokenizer from {params.tokenizer_path}...')
tokenizer = PreTrainedTokenizerFast.from_pretrained(params.tokenizer_path, params.max_len)

print(f'Load model from {params.model_path}...')
device = torch.device('cuda')
model = GPT2LMHeadModel.from_pretrained(params.model_path)
model.to(device)

generated = []

print(f'Generation...')
for input_str in tqdm(input_str_list):
    input_ids = torch.tensor(tokenizer.encode(input_str, add_special_tokens=True)).unsqueeze(0).to(device)
    output_sequences = model.generate(
        input_ids,
        max_length=params.max_len,
#         num_beams=1,
        do_sample=True,
        top_k = params.top_k,
        top_p = params.top_p,
        temperature = params.temperature,
        #no_repeat_ngram_size=params.no_repeat_ngram_size,
        num_return_sequences=1,
        pad_token_id=0,
        early_stopping=False
    )
    generated.append(output_sequences)
print(f'Success, save to ./generated/{params.name}.pkl')    
    
with open(f'./generated/{params.name}.pkl','wb') as fw:
    pickle.dump([generated,params],fw)
