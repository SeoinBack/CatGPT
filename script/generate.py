import sys
sys.path.append('../')

import os
import pickle
import torch
import numpy as np

from catgpt.utils.generation_utils import str_to_atoms
from catgpt.utils.validation_utils import Crystal

from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from tqdm import tqdm
from ase.atoms import Atoms

from omegaconf import OmegaConf

params = OmegaConf.load('../config/generation_config.yaml')
device = torch.device(params.device)

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    f'../tokenizer/{params.string_type}-tokenizer/',
    max_len=1024
)
model = GPT2LMHeadModel.from_pretrained(params.checkpoint_path)

generated = []
i = 0

with tqdm(total = params.num_generation) as pbar:
    while True:
            input_ids = torch.tensor(tokenizer.encode(params.input_prompt, add_special_tokens=True)).unsqueeze(0).to(device)
            output_sequences = model.generate(
                input_ids,
                max_length=1024,
                #num_beams=1,
                do_sample=True,
                top_k = params.top_k,
                top_p = params.top_p,
                temperature = params.temperature,
                #no_repeat_ngram_size=params.no_repeat_ngram_size,
                num_return_sequences=1,
                pad_token_id=1,
                early_stopping=True
            )
            atoms_str = tokenizer.decode(output_sequences[0]).split('. <eos>')[0]

            try:
                atoms, struct_val, gen_val  = str_to_atoms(atoms_str,lat_idx=1,skip_fail=False, early_stop=False)
            except np.linalg.LinAlgError:
                continue
            pbar.update()
        
            if (gen_val == True) & (type(atoms) == Atoms):
                generated.append(output_sequences)
                i += 1
                
            if i > 9999:
                break
                
with open(params.save_path,'wb') as fw:
    pickle.dump(generated,fw)            
