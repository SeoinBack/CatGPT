import sys, os
abs_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(abs_path)

import argparse
import os
import pickle
import torch
import numpy as np

from catgpt.utils.generation_utils import str_to_atoms
from catgpt.utils.validation_utils import Crystal

from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from tqdm import tqdm
from ase.atoms import Atoms


def parse_args():
    parser = argparse.ArgumentParser(description='generate structures from model checkpoint')

    parser.add_argument('--name', type=str, help='name of generated structures', required=True)
    parser.add_argument('--ckpt_path', type=str, help='path to trained generative model checkpoint', required=True)
    parser.add_argument('--save_path', type=str, help='path to generated data to save', required=True)
    parser.add_argument('--string_type', 
                        type=str, 
                        help='tokenization type', 
                        default='coordinate', 
                        choices=['coordinate','digit','split','ads'])
    parser.add_argument('--input_prompt', 
                        type=str, 
                        help='input prompt for generation start with', 
                        default='<bos>',)
    parser.add_argument('--device', type=str, help='device', default='cuda')
    
    # generation parameters
    parser.add_argument('--n_generation', 
                        type=int, 
                        help='the number of structures to generate', 
                        default=10000)
    parser.add_argument('--top_k', 
                        type=int, 
                        help='the number of highest probability vocabulary tokens to keep', 
                        default=30)
    parser.add_argument('--top_p',
                        type=int, 
                        help='most probable tokens set with probabilities that add up to top_p or higher to keep', 
                        default=90)
    parser.add_argument('--temperature', 
                        type=float, 
                        help='value used to modulate the next token probabilities', 
                        default=1.)
    
                        
    
    args = parser.parse_args()

    return args

def generate(args):
    device = torch.device(args.device)
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        f'{abs_path}/data/tokenizer/{args.string_type}-tokenizer/',
        max_len=1024
    )
    model = GPT2LMHeadModel.from_pretrained(args.ckpt_path).to(device)
    
    generated = []
    i = 0
    
    with tqdm(total = args.n_generation) as pbar:
        while True:
                input_ids = torch.tensor(tokenizer.encode(args.input_prompt, add_special_tokens=True)).unsqueeze(0).to(device)
                output_sequences = model.generate(
                    input_ids,
                    max_length=1024,
                    #num_beams=1,
                    do_sample=True,
                    top_k = args.top_k,
                    top_p = args.top_p,
                    temperature = args.temperature,
                    #no_repeat_ngram_size=params.no_repeat_ngram_size,
                    num_return_sequences=1,
                    pad_token_id=1,
                )
                atoms_str = tokenizer.decode(output_sequences[0]).split('. <eos>')[0]
    
                try:
                    atoms, struct_val, gen_val  = str_to_atoms(atoms_str,lat_idx=1, skip_fail=False, early_stop=False)
                except np.linalg.LinAlgError:
                    continue
                pbar.update()
            
                if (gen_val == True) & (type(atoms) == Atoms):
                    generated.append(output_sequences)
                    i += 1
                    
                if i > args.n_generation - 1:
                    break
                    
    with open(args.save_path + '/' + args.name + '.pkl','wb') as fw:
        pickle.dump(generated,fw)            

if __name__ == '__main__':
    args = parse_args()
    generate(args) 