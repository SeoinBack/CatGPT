import sys, os
abs_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(abs_path)

import argparse
import pickle
import torch
import numpy as np

from catgpt.utils.generation_utils import str_to_atoms
from catgpt.utils.validation_utils import Crystal, GenEval

from transformers import BertForSequenceClassification, PreTrainedTokenizerFast
from ase.atoms import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='generate structures from model checkpoint')

    parser.add_argument('--cls-path', type=str, help='path to trained detection model checkpoint', required=True)
    parser.add_argument('--gen-path', type=str, help='path to generated structures', required=True)
    parser.add_argument('--save-path', type=str, help='path to validated data to save', required=True)
    parser.add_argument('--gt-path', 
                        type=str, 
                        help='path to ground-truth structure data', 
                        default=f'{abs_path}/data/OC20-val-10K-crys.pkl', 
                        )    
    parser.add_argument('--string-type', 
                        type=str, 
                        help='tokenization type', 
                        default='coordinate', 
                        choices=['coordinate','digit','split','ads'])
    parser.add_argument('--device', type=str, help='device', default='cuda')
    
    # validation parameters
    parser.add_argument('--n-samples', 
                        type=int,
                        help='the number of structures to validate', 
                        default=5000)
    parser.add_argument('--skip-fail', 
                        action=argparse.BooleanOptionalAction, 
                        help='skip overlapping atoms', 
                        default=False)
                        
    args = parser.parse_args()

    return args
    
def validate(args):
    device = torch.device(args.device)
    
    with open(args.gen_path,'rb') as fr:
        generated = pickle.load(fr)
    with open(args.gt_path,'rb') as fr:
        gt_crys = pickle.load(fr)
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        f'{abs_path}/data/tokenizer/{args.string_type}-tokenizer/',
        max_len=1024
    )
    classifier = BertForSequenceClassification.from_pretrained(args.cls_path).to(device)
    
    generated_atoms = []
    
    
    for gen in tqdm(generated):
        atoms_str = tokenizer.decode(gen[0]).split('. <eos>')[0]
        
        atoms, _, _ = str_to_atoms(
            atoms_str,
            lat_idx=1,
            skip_fail=args.skip_fail,
            )
        
        generated_atoms.append(atoms)
        
    valid_crys = []
    
    for atoms in tqdm(generated_atoms, position=0, leave=True):
        if  (type(atoms) == Atoms) & (atoms.get_volume()>1) & all(atoms.get_cell_lengths_and_angles() > 0.5):
            try:
                valid_crys.append(Crystal(AseAtomsAdaptor.get_structure(atoms), classifier, tokenizer))
            except ValueError:
                continue
                
                
    with open(args.save_path,'wb') as fw:
        pickle.dump(valid_crys,fw)
    
        
    valid_crys = [i for i in valid_crys if i.comp_fp != None]
    gt_crys = [i for i in gt_crys if i.comp_fp != None]
    
    print(GenEval(valid_crys, gt_crys,n_samples=args.n_samples).get_metrics())
    
if __name__ == '__main__':
    args = parse_args()
    validate(args) 
