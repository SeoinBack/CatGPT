import sys, os
abs_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(abs_path)

import pickle
import torch
import numpy as np

from catgpt.generation_utils import str_to_atoms
from catgpt.validation_utils import Crystal, GenEval

from transformers import BertForSequenceClassification, PreTrainedTokenizerFast
from ase.atoms import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from tqdm import tqdm
from omegaconf import OmegaConf

params = OmegaConf.load(abs_path + '/config/validation_config.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(params.generated_path,'rb') as fr:
    generated = pickle.load(fr)
with open(params.ground_truth_path,'rb') as fr:
    gt_crys = pickle.load(fr)

tokenizer = PreTrainedTokenizerFast.from_pretrained(params.tokenizer_path)
classifier = BertForSequenceClassification.from_pretrained(params.classifier_path).to(device)

generated_atoms = []


for gen in tqdm(generated):
    atoms_str = tokenizer.decode(gen[0]).split('. <eos>')[0]
    
    atoms, _, _ = str_to_atoms(
        atoms_str,
        lat_idx=1,
        skip_fail=params.skip_fail,
        early_stop=params.early_stop,
        return_str=params.return_str
        )
    
    generated_atoms.append(atoms)
    
valid_crys = []

for atoms in tqdm(generated_atoms, position=0, leave=True):
    if  (type(atoms) == Atoms) & (atoms.get_volume()>1) & all(atoms.get_cell_lengths_and_angles() > 0.5):
        try:
            valid_crys.append(Crystal(AseAtomsAdaptor.get_structure(atoms), classifier, tokenizer))
        except ValueError:
            continue
            
            
with open(params.save_path,'wb') as fw:
    pickle.dump(valid_crys,fw)

    
valid_crys = [i for i in valid_crys if i.comp_fp != None]
gt_crys = [i for i in gt_crys if i.comp_fp != None]

print(GenEval(valid_crys, gt_crys,n_samples=params.n_samples).get_metrics())